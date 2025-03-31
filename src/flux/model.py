from dataclasses import dataclass

import time, copy
import concurrent.futures
import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams, custom_offload: bool = False, exec_device = 'cuda'):
        super().__init__()

        self.params = params
        self.custom_offload = custom_offload
        self.exec_device = exec_device
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    index=i,
                )
                for i in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False

    def estimate_model_size(self, model: torch.nn.Module, dtype=torch.float32):
        param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * dtype.itemsize
        buffer_size = sum(b.numel() for b in model.buffers()) * dtype.itemsize
        total_size = param_size + buffer_size  # 计算参数+缓冲区的总大小

        return param_size, buffer_size, total_size

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        block_controlnet_hidden_states=None,
        guidance: Tensor | None = None,
        image_proj: Tensor | None = None, 
        ip_scale: Tensor | float = 1.0, 
    ) -> Tensor:
        # param_size, buffer_size, total_size = self.estimate_model_size(self)
        # print(f'forward, size: {param_size/1024/1024}MB, {buffer_size/1024/1024}MB, {total_size/1024/1024}MB')
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        forward_start = time.time()

        # 将blocks以外的module、参数等都转移到GPU
        if self.custom_offload:
            for name, model in self.named_children():
                if name in ['single_blocks', 'double_blocks']:
                    continue
                model.to(self.exec_device)

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # img.to("cuda")
        # txt.to("cuda")
        # vec.to("cuda")
        # if pe is not None:
        #     pe.to("cuda")
        # if image_proj is not None:
        #     image_proj.to("cuda")
        # if ip_scale is not None and isinstance(ip_scale, Tensor):
        #     ip_scale.to("cuda")

        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)

        prepare_done = time.time()

        # 创建和double_blocks数量一样的stream
        load_streams = [torch.cuda.Stream() for _ in range(len(self.double_blocks))]
        # 创建新的double_blocks
        new_double_blocks = nn.ModuleList()
        # 先加载1个double_block
        with torch.cuda.stream(load_streams[0]):
            origin_block = self.double_blocks[0]
            new_block = copy.deepcopy(origin_block).to(self.exec_device, non_blocking=True)
            new_double_blocks.append(new_block)

        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                    image_proj,
                    ip_scale,
                )
            else:
                # 使用stream加载下N个block
                for i in range(1):
                    next_index = index_block + 1 + i
                    if next_index < len(self.double_blocks):
                        with torch.cuda.stream(load_streams[next_index]):
                            block = self.double_blocks[next_index].to("cuda", non_blocking=True)
                            new_double_blocks.append(block)
                # 等待自己的block加载完成
                load_streams[index_block].synchronize()
                # 计算
                block = new_double_blocks[index_block]
                img, txt = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    image_proj=image_proj,
                    ip_scale=ip_scale, 
                )
                # 直接删除block，而不是to cpu
                del block
                torch.cuda.empty_cache()
            # controlnet residual
            if block_controlnet_hidden_states is not None:
                img = img + block_controlnet_hidden_states[index_block % 2]
            cur = time.time()

        double_blocks_done = time.time()
        del new_double_blocks
        torch.cuda.empty_cache()

        img = torch.cat((txt, img), 1)
        # 创建和single_blocks数量一样的stream
        load_streams = [torch.cuda.Stream() for _ in range(len(self.single_blocks))]
        # 创建新的single_blocks
        new_single_blocks = nn.ModuleList()
        # 先加载1个single_block
        t1 = time.time()
        with torch.cuda.stream(load_streams[0]):
            block = self.single_blocks[0].to("cuda", non_blocking=True)
            new_single_blocks.append(block)
        # print(f'load 1 single block, use {time.time()-t1}, total {len(self.single_blocks)} single blocks')
        last_t = time.time()
        for index_block, block in enumerate(self.single_blocks):
            param_size, buffer_size, total_size = self.estimate_model_size(block)
            # print(f'single block, size: {param_size/1024/1024}MB, {buffer_size/1024/1024}MB, {total_size/1024/1024}MB')
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    vec,
                    pe,
                )
            else:
                t1 = time.time()
                # 使用stream加载下一个block
                for i in range(1):
                    next_index = index_block + 1 + i
                    if next_index < len(self.single_blocks):
                        with torch.cuda.stream(load_streams[next_index]):
                            block = self.single_blocks[next_index].to("cuda", non_blocking=True)
                            new_single_blocks.append(block)
                t2 = time.time()
                # 等待自己的block加载完成
                load_streams[index_block].synchronize()

                # # 加载自己的block
                # block.to("cuda")
                t3 = time.time()
                block = new_single_blocks[index_block]
                img = block(img, vec=vec, pe=pe)
                t4 = time.time()
                # with torch.cuda.stream(to_cpu_stream):
                    # block.to("cpu", non_blocking=True)
                del block
                torch.cuda.empty_cache()
                t5 = time.time()

            cur = time.time()
            # print(f'single block index: {index_block}, use {cur-last_t}, next to cuda {t2-t1}, sync wait cur{t3-t2}, calc {t4-t3}, to cpu {t5-t4}')
            last_t = cur
        # print(f'end single block')
        img = img[:, txt.shape[1] :, ...]
        single_blocks_done = time.time()
        del new_single_blocks
        torch.cuda.empty_cache()

        # print(f'start final layer')
        t1 = time.time()
        final_layer = self.final_layer.to("cuda")
        t2 = time.time()
        # print(f'final layer to cuda, use {t2-t1}')
        img = final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        t3 = time.time()
        # print(f'final layer calc, use {t3-t2}')
        forward_end = time.time()
        print(f'flux forward total {forward_end-forward_start}, prepare {prepare_done-forward_start}, double {double_blocks_done-prepare_done}, single {single_blocks_done-double_blocks_done}, final_layer {forward_end - single_blocks_done}')
        return img
