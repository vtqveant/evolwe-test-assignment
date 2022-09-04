from typing import List, Tuple

import csv
import numpy as np
import torch.utils.data
from torch.utils.data.dataset import Dataset
from torch.utils.data import SubsetRandomSampler
from src.label_tracker import LabelTracker


class HelloEvolweDataset(Dataset):
    def __init__(self, filename: str, label_tracker: LabelTracker):
        super(HelloEvolweDataset, self).__init__()
        self.label_tracker = label_tracker
        self.filename = filename
        self.samples = self._load()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "text": sample[0],
            "intent_idx": sample[2]
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self) -> List[Tuple[str, str, int]]:
        samples = []
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            for entry in reader:
                samples.append((
                    entry['text'],
                    entry['intent'],
                    self.label_tracker.get_intent_index(entry['intent'])
                ))
        return samples


def main():
    dataset = HelloEvolweDataset(filename='../data/dataset.csv', label_tracker=LabelTracker())
    for i, entry in enumerate(dataset):
        if i == 100:
            break
        print(entry)

    random_seed = 42
    test_split_portion = 0.1

    n_samples = len(dataset)
    indices = list(range(n_samples))
    split_idx = int(np.floor(test_split_portion * n_samples))

    # shuffle
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split_idx:], indices[:split_idx]

    # samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=test_sampler)

    for sample in train_loader:
        print(sample)

    print(len(train_loader))
    print(len(test_loader))


if __name__ == '__main__':
    main()
