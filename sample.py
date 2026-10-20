from model import Transformer, generate
from dataset import device
import torch
from tokenizer import decode, EOT
from config import CHECKPOINT

model = Transformer()
model = model.to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

model.eval()
context = torch.tensor([[EOT]], dtype=torch.long, device=device)
for _ in range(500):
    context = generate(model, context)

context = context[0].tolist()
print(decode(context))