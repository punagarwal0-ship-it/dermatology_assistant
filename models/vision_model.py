"""
models/vision_model.py

A fine-tuned pretrained vision backbone (via timm) for disease classification.

Default backbone: efficientnet_b0 - small enough to train on a laptop CPU
in reasonable time, while still being a genuinely useful ImageNet-pretrained
feature extractor. Swap via config.yaml / --model without touching this file
as long as the model name is available in timm.

We are NOT training from scratch (per project instructions) - the backbone
starts from ImageNet-pretrained weights and only the head (plus optionally
the last few blocks) is trained initially.
"""

import timm
import torch
import torch.nn as nn


class DermatologyVisionModel(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", num_classes: int = 2,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        if freeze_backbone:
            self._freeze_backbone_except_head()

    def _freeze_backbone_except_head(self):
        """
        Freezes all parameters except the final classifier layer.
        timm models expose the classifier via get_classifier() / the
        model-specific attribute name varies, so we use timm's helper.
        """
        classifier = self.backbone.get_classifier()
        classifier_params = set(id(p) for p in classifier.parameters())

        for p in self.backbone.parameters():
            if id(p) not in classifier_params:
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def get_device(preferred: str = "auto") -> torch.device:
    """
    preferred: "auto" | "cuda" | "cpu"
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] --device cuda requested but CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> DermatologyVisionModel:
    return DermatologyVisionModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
