import torch
import torch.nn as nn
import torchvision
import torch.autograd.profiler as profiler


class FeatureExtractionLayer(nn.Module):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
        origin = torchvision.models.resnet18(pretrained=True)
        self.features = origin

    def forward(self, x):
        result = self.features(x)
        return result
