import sys
import torch
from model import Transformer, generate
from tokenizer import decode, EOT
from config import CHECKPOINT, TEMPERATURE, TOP_K, SAMPLE_TOKENS, SEED, device

# usage: python sample.py [checkpoint]
torch.manual_seed(SEED)
checkpoint = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT

model = Transformer().to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

# start from end-of-text, i.e. "a new story begins here"
context = torch.tensor([[EOT]], dtype=torch.long, device=device)
for _ in range(SAMPLE_TOKENS):
    context = generate(model, context, TEMPERATURE, TOP_K)

print(decode(context[0].tolist()))
