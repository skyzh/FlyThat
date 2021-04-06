from loguru import logger
import torch
from torch import nn
from . import utils
from .dataset import MAX_LABELS, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, MAX_IMAGE_TENSOR


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL *
                      MAX_IMAGE_TENSOR, 5),
            nn.Linear(5, MAX_LABELS),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main(args):
    device = utils.get_device()
    logger.info(f'Using {device} device')
    model = NeuralNetwork().to(device)
    print(model)
