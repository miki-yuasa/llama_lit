from collections.abc import Iterable
from lit_nlp.api.model import JsonDict, Model


class Llama2cModel(Model):
    """
    Wrapper for the llama2c model in .bin format
    """

    def __init__(self, model_path: str) -> None:
        self.model_path: str = model_path

    def predict(self, inputs: Iterable[JsonDict], **kw) -> Iterable[JsonDict]:
        return super().predict(inputs, **kw)
