import pandas as pd
from schemas.pydantic.model_response import AiResponse


from abc import abstractmethod


class Model():
    def __init__(self, name, target):
        self.name = name

    @abstractmethod
    def indigestion(self):
        return False

    @abstractmethod
    def query(self, text_input: str) -> AiResponse:
        pass

    # TODO fix the returning type??
    @abstractmethod
    def classifier(text_inputs: pd.Series):
        pass