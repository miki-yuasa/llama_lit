from lit_nlp import dev_server
import torch

from llama_lit.wrapper.model import Llama2cModel
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

start: str = "Once upon a time, there was a handsome red wolf named Bob."
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(model_path)

model_args = ModelArgs(**checkpoint["model_args"])

model = Transformer(model_args)

model.load_state_dict(checkpoint["model"])
model.to(device=device)

enc = Tokenizer(tokenizer_model=tokenizer_model_path)
start_ids = enc.encode(start, bos=True, eos=False)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(enc.decode(y[0].tolist()))
        print("---------------")

# datasets = {
#     "foo_data": FooDataset("/path/to/foo.tsv"),
#     "bar_data": BarDataset("/path/to/bar.tfrecord"),
# }
# models = {"my_model": Llama2cModel(model_path)}
# lit_demo = dev_server.Server(models, datasets, port=4321)
# lit_demo.serve()

print("Done!")
