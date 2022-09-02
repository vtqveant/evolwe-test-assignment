import yaml
from torch.utils.data.dataset import IterableDataset
from src.label_tracker import LabelTracker, DictLabelTracker


class HelloEvolweDataset(IterableDataset):
    def __init__(self, filename: str, label_tracker: LabelTracker):
        super(HelloEvolweDataset, self).__init__()
        self.label_tracker = label_tracker
        self.filename = filename
        self.samples = self._load()

    def __iter__(self):
        for i, row in enumerate(self.samples):
            yield {
                # "id": i,
                "text": row[0],
                # "intent": row[1],
                "intent_idx": self.label_tracker.get_intent_index(row[1])
            }

    def _load(self):
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
