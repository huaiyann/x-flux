import torch
from diffusers import FluxPipeline
from torch import nn
import time, os


class ModelOffloaderV2:
    def __init__(self, model: nn.Module, record_stream: bool = False):
        # move model to pinned memory. keep a model copy in CPU pinned memory.
        for p in model.parameters():
            p.data = p.data.cpu().pin_memory()
        self.param_dict = {p: p.data for p in model.parameters()}
        self.manual_params = []
        self.stream = torch.cuda.Stream()

        def create_pre_hook(next_layer):
            @torch.compiler.disable()
            def pre_hook(module, args):
                # wait for H2D transfer for the current layer to complete
                self.stream.synchronize()

                if next_layer is None:
                    return

                # start H2D transfer for the next layer
                current_stream = torch.cuda.current_stream()
                with torch.cuda.stream(self.stream):
                    for p in next_layer.parameters():
                        p.data = p.data.cuda(non_blocking=True)

                        # p.data is owned by self.stream
                        # only deallocate once current layer finishes.
                        # compared to torch.cuda.current_stream().synchronize(),
                        # this is slightly faster but uses more memory.
                        if record_stream:
                            p.data.record_stream(current_stream)

            return pre_hook

        @torch.compiler.disable()
        def post_hook(module, args, output):
            if not record_stream:
                torch.cuda.current_stream().synchronize()
            for p in module.parameters():
                p.data = self.param_dict[p]

        def traverse(module: nn.Module):
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                for i in range(len(module)):
                    current_layer = module[i]
                    next_layer = module[i + 1] if i + 1 < len(module) else None
                    current_layer.register_forward_pre_hook(create_pre_hook(next_layer))

                    if i == 0:  # manually move first layer params
                        self.manual_params.extend(module[0].parameters())
                    else:  # don't free first layer params after forward
                        current_layer.register_forward_hook(post_hook)

            else:
                for p in module.parameters(recurse=False):
                    self.manual_params.append(p)
                for child in module.children():
                    traverse(child)

        traverse(model)

    def cuda(self):
        for p in self.manual_params:
            p.data = p.data.cuda(non_blocking=True)

    def cpu(self):
        for p in self.manual_params:
            p.data = self.param_dict[p]


# pipe = FluxPipeline.from_pretrained("/home/ubuntu/models/flux.1-dev", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("/data/models/flux1dev", torch_dtype=torch.bfloat16)
pipe.text_encoder_2.cuda() # T5
# ModelOffloaderV2(pipe.text_encoder_2, record_stream=True).cuda()  # T5
ModelOffloaderV2(pipe.transformer, record_stream=True).cuda()
pipe.text_encoder.cuda()  # CLIP
pipe.vae.cuda()

# uncomment to use torch.compile(). 1st iteration will take some time.
# torch._dynamo.config.cache_size_limit = 10000
# pipe.transformer.compile()

for i in range(2):
    prompt = "A cat holding a sign that says hello world"
    start = time.time()
    image = pipe(
        prompt,
        height=1449,
        width=816,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    ind = len(os.listdir("./results"))
    image.save(os.path.join("./results", f"result_{ind}.png"))
    print(f'use {time.time() - start}')
