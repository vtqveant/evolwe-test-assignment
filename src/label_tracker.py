class LabelTracker:
    """A container for labels with lazy registration"""

    def __init__(self):
        self.label_idx = 0
        self.labels = {}

    def get_intent_index(self, label):
        if label not in self.labels.keys():
            self.labels[label] = self.label_idx
            self.label_idx += 1
        return self.labels[label]

    def get_num_labels(self):
        return len(self.labels)
