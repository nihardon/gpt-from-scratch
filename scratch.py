from model import Head
import torch

head = Head(32)

k = torch.randn(1, 10, 32)
new = head.rotate(k)

mag1 = k.norm(dim=-1)
mag2 = new.norm(dim=-1)

#1
print(torch.allclose(mag1, mag2))

#2
print(torch.allclose(k[0, 0], new[0, 0]))

#3
q0 = torch.randn(32)
q = q0.expand(256, -1)
q = q.unsqueeze(0)

k0 = torch.randn(32)
k = k0.expand(256, -1)
k = k.unsqueeze(0)

q = head.rotate(q)
k = head.rotate(k)

scores = q[0] @ k[0].transpose(-2, -1)
print(scores[10, 7])
print(scores[110, 107])
print(scores[200, 197])
print(scores[10, 5])