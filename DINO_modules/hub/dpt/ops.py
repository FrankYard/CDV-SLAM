# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import warnings
import torch
import torch.nn.functional as F


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def remove_borders(score_map: torch.Tensor, borders: int):
    '''
    It removes the borders of the image to avoid detections on the corners
    '''
    shape = score_map.shape
    mask = torch.ones_like(score_map)

    mask[:, :, 0:borders, :] = 0
    mask[:, :, :, 0:borders] = 0
    mask[:, :, shape[2] - borders:shape[2], :] = 0
    mask[:, :, :, shape[3] - borders:shape[3]] = 0

    return mask * score_map

def sample_descriptors(keypoints, descriptors, sx, sy):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    assert len(keypoints.shape) == 3
    assert keypoints.shape[0] == b and keypoints.shape[-1] == 2
    offset = torch.tensor([sx / 2 + 0.5, sy / 2 + 0.5]).to(keypoints)
    scale = torch.tensor([(w*sx - sx/2 - 0.5), (h*sy - sy/2 - 0.5)]).to(keypoints)
    keypoints = (keypoints - offset) / scale
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

def sample_depth(keypoints, depth, sx, sy):
    b, c, h, w = depth.shape
    assert c == 1
    offset = torch.tensor([sx / 2 + 0.5, sy / 2 + 0.5]).to(keypoints)
    scale = torch.tensor([(w*sx - sx/2 - 0.5), (h*sy - sy/2 - 0.5)]).to(keypoints)
    keypoints = (keypoints - offset) / scale * 2 - 1
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    depth = torch.nn.functional.grid_sample(
        depth, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    depth = depth.reshape(b, 1, -1)
    return depth
