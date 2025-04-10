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

    def __init__(self, params: FluxParams, custom_offload: bool = False, exec_device = None):
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
        self.param_dict = dict()
        self.first = True

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

    # @torch.compile
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

        # if self.first:
        #     # 将blocks以外的module、参数等都转移到GPU
        #     if self.custom_offload:
        #         for name, model in self.named_children():
        #             if name in ['single_blocks', 'double_blocks']:
        #                 continue
        #             model.to(self.exec_device)
        #     # 将blocks[0]放cuda，其他放cpu并明确为pin_memory
        #     for idx, block in enumerate(self.double_blocks):
        #         if idx == 0:
        #             block.to(self.exec_device)
        #             continue
        #         for  p in block.parameters():
        #             p.data = p.data.cpu().pin_memory()
        #             self.param_dict[p] = p.data
        #     for idx, block in enumerate(self.single_blocks):
        #         if idx == 0:
        #             block.to(self.exec_device)
        #             continue
        #         for p in block.parameters():
        #             p.data = p.data.cpu().pin_memory()
        #             self.param_dict[p] = p.data
        #     self.first = False
    
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

        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)

        prepare_done = time.time()

        load_stream = torch.cuda.Stream()


        def load_block(origin_blocks, index, cur_stream):
            if index >= len(origin_blocks):
                return
            with torch.cuda.stream(load_stream):
                block = origin_blocks[index]
                # block = block.to(self.exec_device, non_blocking=True) # 注意这种方式比下面的慢，猜测是没利用上pin_memory
                for p in block.parameters():
                    p.data = p.data.to(self.exec_device, non_blocking=True)
                    # p.data.record_stream(cur_stream) # 说是防止显存错误地提前释放？

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     with_stack=True
        # ) as prof:
        double_next_to_cuda = 0
        double_sync_wait_cur = 0
        double_do_calc = 0
        double_do_offload = 0
        for index_block, block in enumerate(self.double_blocks):
            t1 = time.time()
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
                # 等待自己的block加载完成
                # load_stream.synchronize()
                t2 = time.time()
                # 使用stream加载下N个block
                # current_stream = torch.cuda.current_stream()
                # load_block(self.double_blocks, index_block + 1, cur_stream=current_stream)
                t3 = time.time()
                img, txt = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    image_proj=image_proj,
                    ip_scale=ip_scale, 
                )
                t4 = time.time()
                # if index_block > 0:
                #     for p in block.parameters():
                #         # p.data.to('meta')
                #         p.data = self.param_dict[p]
                t5 = time.time()
                double_next_to_cuda += t3 - t2
                double_sync_wait_cur += t2 - t1
                double_do_calc += t4 - t3
                double_do_offload += t5 - t4
                # print(f'{double_next_to_cuda, double_sync_wait_cur, double_do_calc, double_do_offload}')
            # controlnet residual
            if block_controlnet_hidden_states is not None:
                img = img + block_controlnet_hidden_states[index_block % 2]
        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        double_blocks_done = time.time()

        img = torch.cat((txt, img), 1)

        next_to_cuda = 0
        sync_wait_cur = 0
        do_calc = 0
        do_offload = 0
        for index_block, block in enumerate(self.single_blocks):
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
                # 等待自己的block加载完成
                # load_stream.synchronize()
                t2 = time.time()
                # current_stream = torch.cuda.current_stream()
                # 使用stream加载下N个block
                # load_block(self.single_blocks, index_block+1, cur_stream=current_stream)
                t3 = time.time()
                # 计算
                img = block(img, vec=vec, pe=pe)
                t4 = time.time()
                # if index_block > 0:
                #     for p in block.parameters():
                #         p.data = self.param_dict[p]
                t5 = time.time()
                next_to_cuda += t3 - t2
                sync_wait_cur += t2 - t1
                do_calc += t4 - t3
                do_offload += t5 - t4
        img = img[:, txt.shape[1] :, ...]
        # torch.cuda.empty_cache()
        single_blocks_done = time.time()

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        forward_end = time.time()
        print(f'flux forward total {forward_end-forward_start}, prepare {prepare_done-forward_start}, \
double {double_blocks_done-prepare_done} ({double_next_to_cuda}/{double_sync_wait_cur}/{double_do_calc}/{double_do_offload}), \
single {single_blocks_done-double_blocks_done} ({next_to_cuda}/{sync_wait_cur}/{do_calc}/{do_offload}), \
final_layer {forward_end - single_blocks_done}')
        return img

# 1. 对data使用pin memory，并在to cuda时对data单独执行，减少to cuda时间
# 2. 保存cpu data的指针，to cpu时只是复制回这个指针，不是真的to cpu
# 3. 第0个放gpu上
# 4. 不empty cache，但是要解决显存释放问题，否则基本每两轮会显存满，触发显存清理导致变慢
# 5. 针对4的解决方案是什么？使用1个load stream而不是多个后有所缓解，之前是两次迭代触发一次，现在是10次迭代左右触发一次。生图时间40->32秒