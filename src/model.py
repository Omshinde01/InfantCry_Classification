"""
model.py
========

Model definition for Infant Cry Classification.
"""

import torch.nn as nn
import torchvision.models as models

from config import (
    MODEL_NAME,
    NUM_CLASSES,
    DROPOUT,
    PRETRAINED
)


class InfantCryResNet(nn.Module):
    """
    ResNet model for infant cry classification.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        pretrained: bool = PRETRAINED,
    ):
        super().__init__()

        self.backbone = self._build_backbone(
            model_name,
            pretrained
        )

        self._modify_first_layer()

        self._modify_classifier(
            num_classes,
            dropout
        )

    def _build_backbone(
        self,
        model_name: str,
        pretrained: bool
    ):

        weights = None

        if pretrained:

            if model_name == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1

            elif model_name == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1

            elif model_name == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V2

        if model_name == "resnet18":
            return models.resnet18(weights=weights)

        elif model_name == "resnet34":
            return models.resnet34(weights=weights)

        elif model_name == "resnet50":
            return models.resnet50(weights=weights)

        raise ValueError(f"Unsupported model : {model_name}")

    def _modify_first_layer(self):
        """
        Convert RGB input to single-channel input.
        """

        original = self.backbone.conv1

        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )

    def _modify_classifier(
        self,
        num_classes,
        dropout
    ):
        """
        Replace ImageNet classifier.
        """

        features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                features,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)