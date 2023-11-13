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
        self,
        model_path: str,
        tokenizer_model_path: str,
        num_samples: int = 1,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 300,
    ) -> None:
        self.model_path: str = model_path
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(model_path)
        model_args = ModelArgs(**checkpoint["model_args"])
        self.model = Transformer(model_args)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device=self.device)

        self.enc = Tokenizer(tokenizer_model=tokenizer_model_path)

        self.num_samples: int = num_samples
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.top_k: int = top_k

    def predict(
        self,
        inputs: Iterable[JsonDict],
    ) -> Iterable[JsonDict]:
        predictions = []

        for input in inputs:
            start = input["story"]
            start_ids = self.enc.encode(start, bos=True, eos=False)
            x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

            with torch.no_grad():
                for k in range(self.num_samples):
                    y = self.model.generate(
                        x,
                        self.max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )
                    predictions.append(
                        {
                            "response": self.enc.decode(y[0].tolist()),
                            "token_ids": y[0].tolist(),
                        }
                    )

        return predictions

    def input_spec(self) -> JsonDict:
        return {
            "story": lit_types.TextSegment(),
            "instruction": lit_types.TextSegment(),
            "summary": lit_types.TextSegment(),
            "source": lit_types.TextSegment(),
        }

    def output_spec(self) -> JsonDict:
        return {
            "response": lit_types.GeneratedText(),
            "token_ids": lit_types.TokenEmbeddings(),
        }
