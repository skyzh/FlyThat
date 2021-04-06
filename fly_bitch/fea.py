import torch
import torch.nn as nn
import torchvision

class fea(nn.Module):
    def __init__(self):
        super(fea, self).__init__()
        origin = torchvision.models.resnet18(pretrained=True)  # output为[m, 1000] 需要加别的层
        
        conv = nn.Conv2d(3, 64, 7, stride = 7, padding = 3, bias = False)
        conv.weight = torch.nn.Parameter(origin.conv1.weight)
        pre = nn.Sequential(
                        conv,
                        origin.bn1,
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1))
        
        self.features = nn.Sequential(pre, origin.layer1, origin.layer2, origin.layer3, origin.layer4)

    def forward(self, x):
        return self.features(x)
