import torch
from dataset import get_batch
from tokenizer import decode

x, y = get_batch('train')

print(x.shape)
print(y.shape)

print(decode(x[0].tolist()))