# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only polyphase ConvTranspose1d using dense Conv1d.

Each stride phase becomes an output channel; no zero-inserted waveform is
materialized. Unsupported convolution geometries retain their native forward.
"""

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def _polyphase(x, weight, stride, padding=0, output_padding=0, groups=1, bias=None):
    kernel = weight.shape[-1]
    in_channels = weight.shape[0]
    out_per_group = weight.shape[1]
    out_channels = groups * out_per_group
    taps = (kernel + stride - 1) // stride
    weight = F.pad(weight, (0, taps * stride - kernel))
    weight = weight.reshape(groups, in_channels // groups, out_per_group, taps, stride)
    weight = weight.permute(0, 2, 4, 1, 3).flip(-1).reshape(out_channels * stride, in_channels // groups, taps)
    result = F.conv1d(x, weight, padding=taps - 1, groups=groups)
    result = result.reshape(x.shape[0], out_channels, stride, -1).transpose(2, 3).flatten(2)
    length = (x.shape[-1] - 1) * stride - 2 * padding + kernel + output_padding
    if padding + length > result.shape[-1]:
        result = F.pad(result, (0, padding + length - result.shape[-1]))
    result = result[..., padding : padding + length]
    if bias is not None:
        result = result + bias[None, :, None]
    return result


def polyphase_conv_transpose1d(module: nn.ConvTranspose1d, x: torch.Tensor, output_size=None) -> torch.Tensor:
    if x.ndim != 3 or output_size is not None or module.dilation != (1,) or module.padding_mode != "zeros":
        return module._h3_native_forward(x, output_size=output_size)
    # Read weight on every invocation: weight_norm hooks/parametrizations,
    # offload and dtype conversions may replace it after installation.
    return _polyphase(
        x, module.weight, module.stride[0], module.padding[0], module.output_padding[0], module.groups, module.bias
    )


def _alias_free_upsample(module, x):
    channels = x.shape[1]
    x = F.pad(x, (module.pad, module.pad), mode="replicate")
    x = module.ratio * _polyphase(x, module.filter.expand(channels, -1, -1), module.stride, groups=channels)
    return x[..., module.pad_left : -module.pad_right]


def install_audio_conv1d(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, nn.ConvTranspose1d) and not hasattr(child, "_h3_native_forward"):
            if child.dilation != (1,) or child.stride[0] <= 1:
                continue
            child._h3_native_forward = child.forward
            child.forward = MethodType(polyphase_conv_transpose1d, child)
            count += 1
        elif type(child).__name__ == "UpSample1d" and not hasattr(child, "_h3_native_forward"):
            # Check the checkpoint's alias-free upsampler contract, not merely
            # its class name. Do not globally monkeypatch functional convolution.
            if not all(hasattr(child, name) for name in ("ratio", "stride", "pad", "pad_left", "pad_right", "filter")):
                continue
            if child.stride != child.ratio or child.stride <= 1 or child.pad_right <= 0:
                continue
            if tuple(child.filter.shape[:2]) != (1, 1) or child.filter.ndim != 3:
                continue
            child._h3_native_forward = child.forward
            child.forward = MethodType(_alias_free_upsample, child)
            count += 1
    return count
