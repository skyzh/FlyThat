import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .data_parser import get_train_data, get_test_data
from .feature_extractor import feature_transformer
from loguru import logger
from pathlib import Path
import numpy as np
import argparse
from torch import nn
import torch
from PIL import Image

MAX_IMAGE_TENSOR = 16
IMG_WIDTH = 320
IMG_HEIGHT = 128
IMG_CHANNEL = 3
MAX_LABELS = 30


def gen_onehot(labels, num_classes):
    labels = torch.LongTensor(labels)
    y_onehot = nn.functional.one_hot(labels, num_classes=num_classes)
    y_onehot = y_onehot.sum(dim=0).float()
    return y_onehot


def process_image_stack(image_path, imgs, image_transform, padding=False):
    tensors = []
    for i in range(MAX_IMAGE_TENSOR):
        if i < len(imgs) or not padding:
            img_path = image_path / imgs[i % len(imgs)]
            img_path = str(img_path)
            tensors.append(
                image_transform(
                    Image.open(img_path)))
        else:
            # TODO: we don't need all three channels
            # Padding remaining tensors with all-zero images
            tensors.append(torch.zeros(
                (IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH)))
    return torch.stack(tensors)


class DrosophilaTrainImageDataset(Dataset):
    def __init__(self, data_path, image_transform, partial):
        """Load dataset

        Args:
            data_path (pathlib.Path): Base data path, should contain `train.csv` and images
            image_transform: Transform function for image
        """
        self.data_csv = get_train_data(data_path / 'train.csv')
        self.image_path = data_path / 'train'
        self.image_transform = image_transform
        self.partial = partial

    def __len__(self):
        return 240 if self.partial else len(self.data_csv)

    def __getitem__(self, idx):
        """Get an item from dataset

        The output is a tuple, where first element if of shape
        `[batch, MAX_IMAGE_TENSOR, 3, 128, 320]`, and the second is `[batch, 30]`.
        """
        labels = self.data_csv.iloc[idx, 1]
        imgs = self.data_csv.iloc[idx, 2]
        # if len(imgs) > MAX_IMAGE_TENSOR:
        #     logger.warning(
        #         f'ignore some of the images of {idx}: {len(imgs)} > {MAX_IMAGE_TENSOR}')
        image_data = process_image_stack(
            self.image_path, imgs, self.image_transform)
        return (image_data, gen_onehot(labels, MAX_LABELS))


class DrosophilaTestImageDataset(Dataset):
    def __init__(self, data_path, image_transform, partial):
        """Load dataset

        Args:
            data_path (pathlib.Path): Base data path, should contain `test_without_label.csv` and images
            image_transform: Transform function for image
        """
        self.data_csv = get_test_data(data_path / 'test_without_label.csv')
        self.image_path = data_path / 'test'
        self.image_transform = image_transform
        self.partial = partial

    def __len__(self):
        return 240 if self.partial else len(self.data_csv)

    def __getitem__(self, idx):
        """Get an item from dataset

        The output is a tuple, where first element if of shape
        `[batch, MAX_IMAGE_TENSOR, 3, 128, 320]``.
        """
        imgs = self.data_csv.iloc[idx, 1]
        name = self.data_csv.iloc[idx, 0]
        # if len(imgs) > MAX_IMAGE_TENSOR:
        #     logger.warning(
        #         f'ignore some of the images of {idx}: {len(imgs)} > {MAX_IMAGE_TENSOR}')
        image_data = process_image_stack(
            self.image_path, imgs, self.image_transform)
        return name, image_data


def main(args):
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Generate PyTorch Dataset')
    parser.add_argument('data_path', type=str, nargs='?', help='Path of unzip data',
                        default=str(Path() / 'data'))
    args = parser.parse_args(args)

    data_path = Path(args.data_path)
    logger.info(f"using data path '{data_path}'")

    dataset = DrosophilaTrainImageDataset(
        Path(data_path), feature_transformer())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    train_features, train_labels = next(iter(dataloader))

    logger.info(f"Feature batch shape: {train_features.size()}")
    logger.info(f"Labels batch shape: {train_labels.size()}")

    img = train_features[0][0]
    logger.info(f'Image shape: {img.size()}')
    img = img.permute(2, 1, 0)
    label = train_labels[0]
    logger.info(f'min={img.min()} max={img.max()}')
    logger.info(f'Image shape (for display): {img.size()}')
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")
