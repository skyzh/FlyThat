import torch
def make_weight_sampler(data_csv):
        count = {}
        N = 0
        for (_, label) in data_csv:
            idx = 0
            seq = 1
            for mask in label.int().numpy().tolist():
                idx |= mask * seq
                seq <<= 1 
            
            try:
                count[idx] += 1
            except:
                count[idx] = 1
            
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

        weights = torch.DoubleTensor(weight_per_bag)
        return torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


def main(argv):
    from .data_parser import get_train_data
    from pathlib import Path
    from loguru import logger
    from .dataset import DrosophilaTrainImageDataset
    from .feature_extractor import feature_transformer

    dataset = DrosophilaTrainImageDataset(
        Path(Path() / 'data'), feature_transformer(), False)
    dataset_length = len(dataset)
    train_length = int(dataset_length * 0.8)
    test_length = dataset_length - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, test_length))
    sampler = make_weight_sampler(test_dataset)
    print(sampler)