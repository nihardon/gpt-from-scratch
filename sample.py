from model import Transformer, generate
from dataset import device
import torch
from tokenizer import decode

model = Transformer()
model = model.to(device)
model.load_state_dict(torch.load('best.pt', map_location=device))

model.eval()
context = torch.tensor([[0]], dtype=torch.long, device=device)
for _ in range(500):
    context = generate(model, context)

context = context[0].tolist()
print(decode(context))