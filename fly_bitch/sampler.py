import torch


def generate_weights(data_csv):
    count = {}
    N = 0
    for (_, label) in data_csv:
        idx = 0
        seq = 1
        for mask in label.int().numpy().tolist():
            idx |= mask * seq
            seq <<= 1

        count[idx] = count.get(idx, 0) + 1

        N = N + 1

    weight_per_labelc = {}
    for key, value in count.items():
        weight_per_labelc[key] = len(count) / float(value)

    weight_per_bag = [0] * N
    for i, (_, label) in enumerate(data_csv):
        idx = 0
        seq = 1
        for mask in label.int().numpy().tolist():
            idx |= mask * seq
            seq <<= 1

        weight_per_bag[i] = weight_per_labelc[idx]

    return count, torch.DoubleTensor(weight_per_bag)


def make_weight_sampler(data_csv):
    _, weights = generate_weights(data_csv)
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


def main(_argv):
    from pathlib import Path
    from .dataset import DrosophilaTrainImageDataset
    from .feature_extractor import feature_transformer

    dataset = DrosophilaTrainImageDataset(
        Path() / 'data', feature_transformer(), False)
    dataset_length = len(dataset)
    train_length = int(dataset_length * 0.8)
    test_length = dataset_length - train_length
    train_dataset, _ = torch.utils.data.random_split(
        dataset, (train_length, test_length))
    count, _ = generate_weights(train_dataset)
    from matplotlib import pyplot as plt
    plt.bar(list(count.keys()), count.values(), color='g')
    plt.show()

    sampler = make_weight_sampler(train_dataset)

    for idx, data in enumerate(sampler):
        if idx > 100:
            break
        print(data)
