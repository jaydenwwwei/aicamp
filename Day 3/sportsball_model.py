"""MobileNetV3 classifier used by the sports-ball odd-one-out game."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SportsBallModel(nn.Module):
    """A small transfer-learning image classifier."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.network = mobilenet_v3_small(weights=weights)
        input_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(input_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def freeze_backbone(self) -> None:
        """Train only the classifier during the short first-launch warm-up."""
        for parameter in self.network.features.parameters():
            parameter.requires_grad = False
        for parameter in self.network.classifier.parameters():
            parameter.requires_grad = True

    def enable_online_learning(self) -> None:
        """Fine-tune the classifier and final feature block during game rounds."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.network.features[-1].parameters():
            parameter.requires_grad = True
        for parameter in self.network.classifier.parameters():
            parameter.requires_grad = True
