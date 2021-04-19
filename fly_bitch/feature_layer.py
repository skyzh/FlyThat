import torch
import torch.nn as nn
import torchvision


class FeatureExtractionLayer(nn.Module):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
        origin = torchvision.models.resnet18(pretrained=True)

        self.features = nn.Sequential(
            *list(origin.children())[:-2]
        )

    def forward(self, x):
        return self.features(x)
