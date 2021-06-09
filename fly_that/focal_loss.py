import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Assert: The inputs have been applied on Sigmoid !!!
        # It is equivalent to nn.BCELoss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # pt = [Π(inputs[i]^targets[i]) * ((1-inputs[i])^(1-targets[i]))]^(1/n)
        # BCE_loss = -log(pt)
        pt = torch.exp(-bce_loss)
        # F_loss = α * ((1-pt) ^ γ) * -log(pt)
        F_loss = self.alpha * ((1-pt)**self.gamma) * bce_loss
        return torch.mean(F_loss)


def main(argv):
    from torch import autograd
    inputs = autograd.Variable(torch.randn(3, 3), requires_grad=True)
    m = nn.Sigmoid()
    print(m(inputs))
    target = torch.FloatTensor([[0, 1, 1], [1, 1, 1], [0, 0, 0]])
    loss_fn = FocalLoss()
    loss = loss_fn(m(inputs), target)
    print(loss)
    loss.backward()
