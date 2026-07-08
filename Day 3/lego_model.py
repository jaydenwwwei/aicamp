"""Shared neural-network definition for training and prediction."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


DIRECTION_CLASSES = [
    "front",
    "front-left",
    "left",
    "back-left",
    "back",
    "back-right",
    "right",
    "front-right",
]

COLOR_CLASSES = [
    "yellow",
    "tan",
    "brown",
    "red",
    "blue",
    "green",
    "black",
    "white",
    "gray",
    "other",
]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LegoMultiTaskModel(nn.Module):
    """MobileNetV3 backbone with direction and head-color output heads."""

    def __init__(self, num_directions: int, num_colors: int, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        feature_count = base.classifier[0].in_features
        hidden_count = base.classifier[0].out_features
        self.shared = nn.Sequential(
            nn.Linear(feature_count, hidden_count),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
        )
        self.direction_head = nn.Linear(hidden_count, num_directions)
        self.color_head = nn.Linear(hidden_count, num_colors)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        shared = self.shared(features)
        return self.direction_head(shared), self.color_head(shared)

    def freeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = True
