from collections.abc import Iterable
from lit_nlp.api.model import JsonDict, Model
import lit_nlp.api.types as lit_types
import torch

from model import ModelArgs, Transformer
from tokenizer import Tokenizer


class Llama2cModel(Model):
    """
    Wrapper for the llama2c model in .bin format
    """

    def __init__(
        self, model_path: str, tokenizer_model_path: str, device: torch.device
    ) -> None:
        self.model_path: str = model_path
        self.device: torch.device = device

        checkpoint = torch.load(model_path)
        model_args = ModelArgs(**checkpoint["model_args"])
        self.model = Transformer(model_args)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device=device)

        self.enc = Tokenizer(tokenizer_model=tokenizer_model_path)

    def predict(
        self,
        inputs: Iterable[JsonDict],
        num_samples: int = 1,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 300,
    ) -> Iterable[JsonDict]:
        predictions = []

        for input in inputs:
            start = input["story"]
            start_ids = self.enc.encode(start, bos=True, eos=False)
            x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

            with torch.no_grad():
                for k in range(num_samples):
                    y = self.model.generate(
                        x, max_new_tokens, temperature=temperature, top_k=top_k
                    )
                    predictions.append({"response": self.enc.decode(y[0].tolist())})

        return predictions

    def input_spec(self) -> JsonDict:
        return {
            "story": lit_types.TextSegment(),
            "instruction": lit_types.TextSegment(),
            "summary": lit_types.TextSegment(),
            "source": lit_types.TextSegment(),
        }

    def output_spec(self) -> JsonDict:
        return {"response": lit_types.TextSegment()}
