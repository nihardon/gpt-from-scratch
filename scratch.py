import torch
import torch.nn.functional as F
from torch import nn

T = 4
C = 8
head_size = 4

x = torch.randn(T, C)


query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

q = query(x)
k = key(x)
v = value(x)

k = k.transpose(-2, -1)

scores = q @ k
scores = scores / (head_size ** 0.5)

mask = torch.ones(T, T)
mask = torch.tril(mask)

scores = scores.masked_fill(mask == 0, float('-inf'))
scores = F.softmax(scores, dim = -1)

out = scores @ v


a = torch.randn(1000, 64)
b = torch.randn(1000, 64)
b = b.transpose(-2, -1)
grid = a @ b
print(grid.std())
grid = grid/8
print(grid.std())
