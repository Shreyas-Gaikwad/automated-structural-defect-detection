import torch.nn as nn
from torchvision import models


class CrackClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        from torchvision.models import resnet50, ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

    def forward(self, x):
        return self.backbone(x)