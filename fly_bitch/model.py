from loguru import logger
import torch
from torch import nn
from . import utils
from .dataset import MAX_LABELS, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, MAX_IMAGE_TENSOR
from . import aggregate_layer
from .feature_layer import FeatureExtractionLayer


class NeuralNetwork(nn.Module):
    def __init__(self, logging=False):
        super(NeuralNetwork, self).__init__()
        self.feature = FeatureExtractionLayer()
        self.aggregate = aggregate_layer.Agg("L1", logging=logging)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20480, MAX_LABELS),
            nn.Softmax(dim=-1)
        )
        self.logging = logging

    def forward(self, x):
        if self.logging:
            logger.info(f'Input shape: {x.shape}')

        # Apply Conv2D to all elements in all batches
        batch_size, ninstance, c, h, w = x.shape
        x = x.reshape(batch_size * ninstance, c, h, w)
        x = self.feature(x)
        if self.logging:
            logger.info(f'After feature extraction: {x.shape}')

        # Aggregate on each batch
        _, c, h, w = x.shape
        x = x.reshape(batch_size, ninstance, c, h, w)
        x = self.aggregate(x)
        if self.logging:
            logger.info(f'After feature aggregate: {x.shape}')

        x = self.flatten(x)
        if self.logging:
            logger.info(f'After flatten: {x.shape}')

        x = self.linear_relu_stack(x)

        if self.logging:
            logger.info(f'Output: {x.shape}')

        return x


def main(args):
    device = utils.get_device()
    logger.info(f'Using {device} device')
    model = NeuralNetwork().to(device)
    print(model)
