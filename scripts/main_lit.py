import json

from lit_nlp import dev_server
import torch

from llama_lit.wrapper.model import Llama2cModel
from llama_lit.wrapper.dataset import Llama2cDataset

num_samples = 1  # number of samples to draw
max_new_tokens = 100  # number of tokens generated in each sample
temperature = (
    1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    300  # retain only the top_k most likely tokens, clamp others to have 0 probability
)

model_path: str = "stories15M.pt"
tokenizer_model_path: str = "tokenizer.model"

dataset_path: str = "data/TinyStories_all_data/data00.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Llama2cDataset(dataset_path)
model = Llama2cModel(
    model_path,
    tokenizer_model_path,
    device,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
)

lit_demo = dev_server.Server(model, dataset, port=4321)
lit_demo.serve()
