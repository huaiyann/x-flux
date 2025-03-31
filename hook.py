import contextlib
import gc
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union, Mapping

import torch
import torch.nn as nn

from accelerate.utils import find_device, send_to_device
from accelerate.utils.imports import (
    is_mlu_available,
    is_mps_available,
    is_musa_available,
    is_npu_available,
    is_peft_available,
    is_torch_xla_available,
    is_xpu_available,
)
from accelerate.utils.memory import clear_device_cache, get_xpu_available_memory
from accelerate.utils.offload import load_offloaded_weight, offload_weight, save_offload_index
from accelerate.utils.tqdm import is_tqdm_available, tqdm
from accelerate.utils.versions import compare_versions, is_torch_version


def named_module_tensors(
    module: nn.Module, recurse: bool = False
):
    yield from module.named_parameters(recurse=recurse)

def check_device_same(first_device, second_device):
    """
    Utility method to check if two `torch` devices are similar. When dealing with CUDA devices, torch throws `False`
    for `torch.device("cuda") == torch.device("cuda:0")` whereas they should be the same

    Args:
        first_device (`torch.device`):
            First device to check
        second_device (`torch.device`):
            Second device to check
    """
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        # In case the first_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        first_device = torch.device("cuda", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        # In case the second_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        second_device = torch.device("cuda", index=0)

    return first_device == second_device

def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    if value is not None:
        # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
        # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
        if old_value.shape != value.shape and param_cls.__name__ != "Params4bit":
            raise ValueError(
                f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" (which has shape {old_value.shape}), this looks incorrect.'
            )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    device_quantization = None
    with torch.no_grad():
        # leave it on cpu first before moving them to cuda
        # # fix the case where the device is meta, we don't want to put it on cpu because there is no data =0
        if (
            param is not None
            and param.device.type != "cuda"
            and torch.device(device).type == "cuda"
            and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        ):
            device_quantization = device
            device = "cpu"
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if isinstance(device, int):
            if is_xpu_available():
                device = f"xpu:{device}"
        if "xpu" in str(device) and not is_xpu_available():
            raise ValueError(f'{device} is not available, you should use device="cpu" instead')
        if value is None:
            new_value = old_value.to(device)
            if dtype is not None and device in ["meta", torch.device("meta")]:
                if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype)

                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]:
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    # downcast to fp16 if any - needed for 8bit serialization
                    new_value = new_value.to(torch.float16)
                # quantize module that are going to stay on the cpu so that we offload quantized weights
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
            elif param_cls.__name__ in ["QTensor", "QBitsTensor"]:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(device)
            elif param_cls.__name__ in ["AffineQuantizedTensor"]:
                if importlib.util.find_spec("torchao") is not None and compare_versions("torchao", ">=", "0.7.0"):
                    # TorchAO v0.7.0 made layout_tensor an internal private variable and exposed tensor_impl
                    args = (new_value.tensor_impl,)
                else:
                    args = (new_value.layout_tensor,)
                args += (
                    new_value.block_size,
                    new_value.shape,
                    new_value.quant_min,
                    new_value.quant_max,
                    new_value.zero_point_domain,
                )
                new_value = torch.nn.Parameter(param_cls(*args), requires_grad=old_value.requires_grad).to(device)
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)

            module._parameters[tensor_name] = new_value
            # as we put the weight to meta, it doesn't have SCB attr anymore. make sure that it is not a meta weight
            if (
                module.__class__.__name__ == "Linear8bitLt"
                and getattr(module.weight, "SCB", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "SCB", None) and device_index is not None:
                    if module.bias is not None and module.bias.device.type != "meta":
                        # if a bias exists, we need to wait until the bias is set on the correct device
                        module = module.cuda(device_index)
                    elif module.bias is None:
                        # if no bias exists, we can quantize right away
                        module = module.cuda(device_index)
            elif (
                module.__class__.__name__ == "Linear4bit"
                and getattr(module.weight, "quant_state", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "quant_state", None) and device_index is not None:
                    module.weight = module.weight.cuda(device_index)
    # clean pre and post foward hook
    if device != "cpu":
        clear_device_cache()

def get_non_persistent_buffers(module: nn.Module, recurse: bool = False):
    """
    Gather all non persistent buffers of a given modules into a set

    Args:
        module (`nn.Module`):
            The module we want the non persistent buffers on.
        recurse (`bool`, *optional*, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct non persistent buffers.
    """

    non_persistent_buffers_set = module._non_persistent_buffers_set
    if recurse:
        for _, m in module.named_modules():
            non_persistent_buffers_set |= m._non_persistent_buffers_set

    return non_persistent_buffers_set


class ModelHook:
    """
    A hook that contains callbacks to be executed just before and after the forward method of a model. The difference
    with PyTorch existing hooks is that they get passed along the kwargs.

    Class attribute:
    - **no_grad** (`bool`, *optional*, defaults to `False`) -- Whether or not to execute the actual forward pass under
      the `torch.no_grad()` context manager.
    """

    no_grad = False

    def init_hook(self, module):
        """
        To be executed when the hook is attached to the module.

        Args:
            module (`torch.nn.Module`): The module attached to this hook.
        """
        return module

    def pre_forward(self, module, *args, **kwargs):
        """
        To be executed just before the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass will be executed just after this event.
            args (`Tuple[Any]`): The positional arguments passed to the module.
            kwargs (`Dict[Str, Any]`): The keyword arguments passed to the module.

        Returns:
            `Tuple[Tuple[Any], Dict[Str, Any]]`: A tuple with the treated `args` and `kwargs`.
        """
        return args, kwargs

    def post_forward(self, module, output):
        """
        To be executed just after the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass been executed just before this event.
            output (`Any`): The output of the module.

        Returns:
            `Any`: The processed `output`.
        """
        return output

    def detach_hook(self, module):
        """
        To be executed when the hook is detached from a module.

        Args:
            module (`torch.nn.Module`): The module detached from this hook.
        """
        return module

class AlignDevicesHook(ModelHook):
    """
    A generic `ModelHook` that ensures inputs and model weights are on the same device for the forward pass of the
    associated module, potentially offloading the weights after the forward pass.

    Args:
        execution_device (`torch.device`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass.
        io_same_device (`bool`, *optional*, defaults to `False`):
            Whether or not the output should be placed on the same device as the input was.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        place_submodules (`bool`, *optional*, defaults to `False`):
            Whether to place the submodules on `execution_device` during the `init_hook` event.
    """

    def __init__(
        self,
        execution_device: Optional[Union[int, str, torch.device]] = None,
        offload: bool = False,
        io_same_device: bool = False,
        weights_map: Optional[Mapping] = None,
        offload_buffers: bool = False,
        place_submodules: bool = False,
        skip_keys: Optional[Union[str, List[str]]] = None,
    ):
        self.execution_device = execution_device
        self.offload = offload
        self.io_same_device = io_same_device
        self.weights_map = weights_map
        self.offload_buffers = offload_buffers
        self.place_submodules = place_submodules
        self.skip_keys = skip_keys

        # Will contain the input device when `io_same_device=True`.
        self.input_device = None
        self.param_original_devices = {}
        self.buffer_original_devices = {}

    def __repr__(self):
        return (
            f"AlignDevicesHook(execution_device={self.execution_device}, offload={self.offload}, "
            f"io_same_device={self.io_same_device}, offload_buffers={self.offload_buffers}, "
            f"place_submodules={self.place_submodules}, skip_keys={repr(self.skip_keys)})"
        )

    def init_hook(self, module):
        if self.offload:
            self.original_devices = {
                name: param.device for name, param in named_module_tensors(module, recurse=self.place_submodules)
            }
            if self.weights_map is None:
                self.weights_map = {
                    name: param.to("cpu")
                    for name, param in named_module_tensors(
                        module, include_buffers=self.offload_buffers, recurse=self.place_submodules
                    )
                }
            for name, _ in named_module_tensors(
                module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True
            ):
                set_module_tensor_to_device(module, name, "meta")

            if not self.offload_buffers and self.execution_device is not None:
                for name, _ in module.named_buffers(recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device
                    )
            elif self.offload_buffers and self.execution_device is not None:
                for name in get_non_persistent_buffers(module, recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device
                    )

        return module

    def pre_forward(self, module, *args, **kwargs):
        if self.io_same_device:
            self.input_device = find_device([args, kwargs])
        if self.offload:
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                value = self.weights_map[name]
                set_module_tensor_to_device(
                    module,
                    name,
                    self.execution_device,
                    value=value
                )

        return send_to_device(args, self.execution_device), send_to_device(
            kwargs, self.execution_device, skip_keys=self.skip_keys
        )

    def post_forward(self, module, output):
        if self.offload:
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                set_module_tensor_to_device(module, name, "meta")
        # 把输出送回到输入设备self.input_device
        if self.io_same_device and self.input_device is not None:
            output = send_to_device(output, self.input_device, skip_keys=self.skip_keys)

        return output

    def detach_hook(self, module):
        if self.offload:
            for name, device in self.original_devices.items():
                if device != torch.device("meta"):
                    set_module_tensor_to_device(module, name, device, value=self.weights_map.get(name, None))
        return module
