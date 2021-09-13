import abc

class AbstractClassifier(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, text, labels):
        pass

    @abc.abstractmethod
    def predict(self, text):
        pass
