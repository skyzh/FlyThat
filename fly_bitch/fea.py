import torch
import torch.nn as nn
import torchvision


class fea(nn.Module):
    def __init__(self):
        super(fea, self).__init__()
        origin = torchvision.models.resnet18(
            pretrained=True)  # output为[m, 1000] 需要加别的层

        self.features = nn.Sequential(
            *list(origin.children())[:-1]
        )

    def forward(self, x):
        return self.features(x)
