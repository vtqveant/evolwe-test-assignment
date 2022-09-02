import abc


class LabelTracker(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_intent_index') and callable(subclass.get_intent_index) or NotImplemented)

    @abc.abstractmethod
    def get_intent_index(self, language: str) -> int:
        raise NotImplementedError


class DictLabelTracker(LabelTracker):
    """A container for labels with lazy registration"""

    def __init__(self):
        self.intent_index = 0
        self.intents = {}

    def get_intent_index(self, intent):
        if intent not in self.intents.keys():
            self.intents[intent] = self.intent_index
            self.intent_index += 1
        return self.intents[intent]
