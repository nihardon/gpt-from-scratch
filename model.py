import torch
from torch import nn
import torch.nn.functional as F
from dataset import get_batch, BLOCK_SIZE
from tokenizer import VOCAB_SIZE


N_EMBED = 32

torch.manual_seed(1)

x, y = get_batch('train')


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        mask = torch.ones(BLOCK_SIZE, BLOCK_SIZE)
        mask = torch.tril(mask)
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        k = k.transpose(-2, -1)
        scores = q @ k
        mask = self.mask[:T, :T]
        scores = scores / (self.head_size ** 0.5)
        scores = scores.masked_fill(mask==0, float('-inf'))
        scores = F.softmax(scores, dim = -1)
        scores = scores @ v
        return scores


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(N_EMBED, 4*N_EMBED), 
                                nn.ReLU(), 
                                nn.Linear(4*N_EMBED, N_EMBED))

    def forward(self, x):
        out = self.ff(x)
        return out

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_model = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.positional_encoding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.head = Head(head_size=N_EMBED)
        self.ff = FeedForward()
        self.out_scores = nn.Linear(N_EMBED, VOCAB_SIZE)

    def forward(self, x, y):
        positions = torch.arange(x.shape[1])

        embeddings = self.embedding_model(x)
        positional_encodings = self.positional_encoding(positions)
        stream = embeddings + positional_encodings
        context = self.head(stream)
        stream = stream + context
        stream = stream + self.ff(stream)

        logits = self.out_scores(stream)
        logits = logits.view(x.shape[0] * x.shape[1], VOCAB_SIZE)
        y = y.view(logits.shape[0])
        loss = F.cross_entropy(logits, y)
        return logits, loss


model = GPT()
optimizer = torch.optim.AdamW(model.parameters(), 0.01)
params = 0
for layer in model.parameters():
     params += layer.numel()
print(params)

for step in range(100):
    loss = model(x,y)[1]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(loss.item())