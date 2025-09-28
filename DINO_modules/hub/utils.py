# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


class Padding(nn.Module):
    def __init__(self, multiple, mode='center'):
        super().__init__()
        self.multiple = multiple
        self.mode = mode

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        if self.mode == 'center':
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
        elif self.mode == 'right':
            pad_size_left, pad_size_right = 0, pad_size
        else:
            raise NotImplementedError
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
