from typing import List, Tuple

import csv
from torch.utils.data.dataset import Dataset
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

    def get_class_weights(self):
        n_classes = self.label_tracker.get_num_labels()
        n_samples = [0 for _ in range(n_classes)]
        for sample in self.samples:
            i = self.label_tracker.get_intent_index(sample[1])
            n_samples[i] += 1
        weights = [count / n_classes for count in n_samples]
        return weights

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
    dataset = HelloEvolweDataset(filename='../data/train.csv', label_tracker=LabelTracker())
    for i, entry in enumerate(dataset):
        if i == 100:
            break
        print(entry)

    class_weights = dataset.get_class_weights()
    print(class_weights)

    sum = 0
    for x in class_weights:
        sum += x
    print(sum)


if __name__ == '__main__':
    main()
