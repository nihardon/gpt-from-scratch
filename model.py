import torch
from torch import nn
import torch.nn.functional as F
from dataset import get_batch, BLOCK_SIZE, BATCH_SIZE
from tokenizer import VOCAB_SIZE


N_EMBED = 32

embedding_model = nn.Embedding(VOCAB_SIZE, N_EMBED)
positional_encoding = nn.Embedding(BLOCK_SIZE, N_EMBED)
positions = torch.arange(BLOCK_SIZE)
out_scores = nn.Linear(N_EMBED, VOCAB_SIZE)

optimizer = torch.optim.AdamW(list(embedding_model.parameters()) +
                              list(positional_encoding.parameters()) +
                              list(out_scores.parameters()), 0.01)

x, y = get_batch('train')

for step in range(100):
    embeddings = embedding_model(x)
    positional_encodings = positional_encoding(positions)
    positional_embeddings = embeddings + positional_encodings
    logits = out_scores(positional_embeddings)
    logits = logits.view(BATCH_SIZE * BLOCK_SIZE, VOCAB_SIZE)
    y = y.view(BATCH_SIZE * BLOCK_SIZE)
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(loss.item())

