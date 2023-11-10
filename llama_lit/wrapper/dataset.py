import json

from lit_nlp.api.dataset import Dataset, Spec
import lit_nlp.api.types as lit_types


class Llama2cDataset(Dataset):
    "Loader for the llama2c dataset."

    def __init__(self, path: str) -> None:
        self.path = path

        with open(path, "r") as f:
            self._examples = json.load(f)

    def spec(self) -> Spec:
        return {
            "story": lit_types.TextSegment(),
            "instruction": lit_types.TextSegment(),
            "summary": lit_types.TextSegment(),
            "source": lit_types.TextSegment(),
        }
