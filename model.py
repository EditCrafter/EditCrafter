import math

import torch
import torch.nn.functional as F
from diffusers.models.lora import LoRACompatibleConv
from torch import Tensor


def inflate_kernels(unet, inflate_conv_list, inflation_transform):
    """Replace selected UNet convolutions with inflated-kernel versions.

    Args:
        unet: The UNet model whose convolutions will be replaced.
        inflate_conv_list: List of module names to inflate.
        inflation_transform: Square transform matrix loaded from a .mat file.
    """

    def replace_module(module, name, index=None, value=None):
        if len(name) == 1 and len(index) == 0:
            setattr(module, name[0], value)
            return module
        current_name, next_name = name[0], name[1:]
        current_index, next_index = int(index[0]), index[1:]
        replace = getattr(module, current_name)
        replace[current_index] = replace_module(
            replace[current_index], next_name, next_index, value
        )
        setattr(module, current_name, replace)
        return module

    for name, module in unet.named_modules():
        if name in inflate_conv_list:
            weight, bias = module.weight.detach(), module.bias.detach()
            (i, o, *_), kernel_size = (
                weight.shape,
                int(math.sqrt(inflation_transform.shape[0])),
            )
            transformed_weight = torch.einsum(
                "mn, ion -> iom",
                inflation_transform.to(dtype=weight.dtype),
                weight.view(i, o, -1),
            )
            conv = LoRACompatibleConv(
                o,
                i,
                (kernel_size, kernel_size),
                stride=module.stride,
                padding=module.padding,
                device=weight.device,
                dtype=weight.dtype,
            )
            conv.weight.detach().copy_(
                transformed_weight.view(i, o, kernel_size, kernel_size)
            )
            conv.bias.detach().copy_(bias)

            sub_names = name.split(".")
            if name.startswith("mid_block"):
                names, indexes = sub_names[1::2], sub_names[2::2]
                unet.mid_block = replace_module(unet.mid_block, names, indexes, conv)
            else:
                names, indexes = sub_names[0::2], sub_names[1::2]
                replace_module(unet, names, indexes, conv)


class ReDilateConvProcessor:
    """Re-dilation convolution processor for higher-resolution generation.

    Dynamically adjusts dilation and padding of convolution modules to handle
    resolution scales beyond the model's training resolution.

    Args:
        module: The original Conv2d module.
        pf_factor: Dilation factor (can be fractional).
        mode: Interpolation mode for resizing.
        activate: Whether dilation is active for the current timestep.
    """

    def __init__(self, module, pf_factor=1.0, mode="bilinear", activate=True):
        self.dilation = math.ceil(pf_factor)
        self.factor = float(self.dilation / pf_factor)
        self.module = module
        self.mode = mode
        self.activate = activate

    def __call__(self, input: Tensor, **kwargs) -> Tensor:
        if self.activate:
            ori_dilation, ori_padding = self.module.dilation, self.module.padding
            inflation_kernel_size = (self.module.weight.shape[-1] - 3) // 2
            self.module.dilation, self.module.padding = (
                self.dilation,
                (
                    self.dilation * (1 + inflation_kernel_size),
                    self.dilation * (1 + inflation_kernel_size),
                ),
            )
            ori_size, new_size = (
                (
                    int(input.shape[-2] / self.module.stride[0]),
                    int(input.shape[-1] / self.module.stride[1]),
                ),
                (
                    round(input.shape[-2] * self.factor),
                    round(input.shape[-1] * self.factor),
                ),
            )
            input = F.interpolate(input, size=new_size, mode=self.mode)
            input = self.module._conv_forward(
                input, self.module.weight, self.module.bias
            )
            self.module.dilation, self.module.padding = ori_dilation, ori_padding
            result = F.interpolate(input, size=ori_size, mode=self.mode)
            return result
        else:
            return self.module._conv_forward(
                input, self.module.weight, self.module.bias
            )
