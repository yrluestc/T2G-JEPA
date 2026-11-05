"""Shared deterministic DROID image preprocessing for metric training and eval."""

import torch
from torchvision.transforms import v2 as transforms

from data import IMAGENET_STATS


def _center_square_crop(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    size = min(h, w)
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top : top + size, left : left + size]


def img_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(_center_square_crop),
        transforms.Resize(size=[img_size, img_size], antialias=False),
        transforms.Normalize(**IMAGENET_STATS),
    ])
