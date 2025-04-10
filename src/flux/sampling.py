import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor
import time, copy

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str],
            custom_offload: bool = False, device = None) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]

    t1 = time.time()
    # if custom_offload and device is not None:
    #     # custom_offload: 先在cpu上copy一份，然后转移至GPU，使用后直接del
    #     t5 = copy.deepcopy(t5)
    #     t5 = t5.to(device)
    t2 = time.time()
    txt = t5(prompt)
    t3 = time.time()
    # if custom_offload and device is not None:
    #     del t5
    t4 = time.time()
    print(f'prepare t5 load {t2-t1}, calc {t3-t2}, offload {t4-t3}')

    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    t1 = time.time()
    # if custom_offload and device is not None:
    #     clip = copy.deepcopy(clip)
    #     clip = clip.to(device)
    t2 = time.time()
    vec = clip(prompt)
    t3 = time.time()
    # if custom_offload and device is not None:
    #     del clip
    t4 = time.time()
    print(f'prepare clip load {t2-t1}, calc {t3-t2}, offload {t4-t3}')

    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        start = time.time()
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     with_stack=True
        # ) as prof:
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale, 
        )
        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        middle = time.time()
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec, 
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
        end = time.time()
        print(f'denoise step {i} use {end-start}, prompt use {middle-start}, neg use {end-middle}')
    return img

def denoise_controlnet(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
):
    # this is ignored for schnell
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond,
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
   
        img = img + (t_prev - t_curr) * pred

        i += 1
    return img

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
