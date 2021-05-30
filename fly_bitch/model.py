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
        self.aggregate = aggregate_layer.SimpleAgg(logging=logging)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, MAX_LABELS),
            nn.Sigmoid()
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

        # # Aggregate on each batch
        _, c = x.shape
        x = x.reshape(batch_size, ninstance, c)

        if self.logging:
            logger.info(f'Before feature aggregate: {x.shape}')

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
    from prettytable import PrettyTable
    device = utils.get_device()
    logger.info(f'Using {device} device')
    model = NeuralNetwork().to(device)
    print(model)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
