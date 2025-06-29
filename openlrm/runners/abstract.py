from abc import ABC, abstractmethod


class Runner(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass
