from model import Transformer, generate
from dataset import device
import torch
from tokenizer import decode, EOT
from config import CHECKPOINT, TOP_K, TEMPERATURE, SEED
torch.manual_seed(SEED)



model = Transformer()
model = model.to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

model.eval()
context = torch.tensor([[EOT]], dtype=torch.long, device=device)
for _ in range(500):
    context = generate(model, context, TEMPERATURE, TOP_K)

context = context[0].tolist()
print(decode(context))