from typing import List, Tuple

import yaml
from torch.utils.data.dataset import Dataset
from src.label_tracker import LabelTracker, DictLabelTracker


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
            "intent_idx": self.label_tracker.get_intent_index(sample[1])
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self) -> List[Tuple[str, str]]:
        samples = []
        with open(self.filename, 'r') as file:
            documents = yaml.full_load(file)
            for entry in documents['data']:
                intent = entry['intent']
                for example in entry['examples']:
                    samples.append((example, intent))
        return samples


def main():
    dataset = HelloEvolweDataset(filename='../data/hello_nova_intents_0.2.2.yaml', label_tracker=DictLabelTracker())
    for i, entry in enumerate(dataset):
        if i == 100:
            break
        print(entry)


if __name__ == '__main__':
    main()
