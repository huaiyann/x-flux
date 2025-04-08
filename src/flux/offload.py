import torch
from torch import nn
import time

class ModelOffloader:
    def __init__(self, model: nn.Module, device):
        # move model to pinned memory. keep a model copy in CPU pinned memory.
        for p in model.parameters():
            p.data = p.data.cpu().pin_memory()
        self.param_dict = {p: p.data for p in model.parameters()}
        self.manual_params = [] # 不参与offload的参数
        self.stream = torch.cuda.Stream()
        self.device = device

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
                        p.data = p.data.to(self.device, non_blocking=True)

                        # p.data is owned by self.stream
                        # only deallocate once current layer finishes.
                        p.data.record_stream(current_stream)

            return pre_hook

        @torch.compiler.disable()
        def post_hook(module, args, output):
            for p in module.parameters():
                p.data = self.param_dict[p]

        def traverse(module: nn.Module):
            if isinstance(module, (nn.ModuleList)): # , nn.Sequential
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

    def load(self):
        # 把不参与offload的参数，直接加载到指定的device
        for p in self.manual_params:
            p.data = p.data.to(self.device, non_blocking=True)
