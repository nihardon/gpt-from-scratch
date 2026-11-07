import torch
from torch import nn
import torch.nn.functional as F
from config import BLOCK_SIZE, N_EMBED, N_HEAD, N_LAYER, DROPOUT, NORM, POS
from tokenizer import VOCAB_SIZE


class RMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.epsilon = .000001

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        out = x / rms
        return out * self.scale

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size

        if POS == 'rope':
            # pair i rotates at speed 10000^(-2i/d): fast pairs for nearby tokens, slow for far
            seq = torch.arange(head_size//2) * (-2/head_size)
            speeds = 10000 ** seq
            positions = torch.arange(BLOCK_SIZE)
            angles = torch.outer(positions, speeds)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            self.register_buffer('cos', cos, persistent=False)
            self.register_buffer('sin', sin, persistent=False)

        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        
        mask = torch.ones(BLOCK_SIZE, BLOCK_SIZE)
        mask = torch.tril(mask)
        self.register_buffer('mask', mask)
        
        self.dropout = nn.Dropout(DROPOUT)

    def rotate(self, vec):
        # RoPE: rotate (first half, second half) pairs by position, so q·k depends only on distance
        T = vec.shape[1]
        a = vec[:, :, :self.head_size//2]
        b = vec[:, :, self.head_size//2:]
        cos = self.cos[:T]
        sin = self.sin[:T]
        
        new_a = a * cos - b * sin
        new_b = a * sin + b * cos
        return torch.cat((new_a, new_b), dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        if POS == 'rope':
            q = self.rotate(q)
            k = self.rotate(k)

        k = k.transpose(-2, -1)
        scores = q @ k
        mask = self.mask[:T, :T]
        scores = scores / (self.head_size ** 0.5)
        scores = scores.masked_fill(mask==0, float('-inf'))
        scores = F.softmax(scores, dim = -1)
        scores = self.dropout(scores)
        scores = scores @ v
        return scores

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(N_HEAD):
            self.heads.append(Head(N_EMBED//N_HEAD))
        self.linear_layer = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        out = torch.cat(out, dim=-1)
        out = self.linear_layer(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(N_EMBED, 4*N_EMBED), 
                                nn.ReLU(), 
                                nn.Linear(4*N_EMBED, N_EMBED),
                                nn.Dropout(DROPOUT))

    def forward(self, x):
        out = self.ff(x)
        return out

class Block(nn.Module):    
    def __init__(self):
        super().__init__()
        if NORM == 'rms':
            self.norm1 = RMSNorm(N_EMBED)
            self.norm2 = RMSNorm(N_EMBED)
        else:
            self.norm1 = nn.LayerNorm(N_EMBED)
            self.norm2 = nn.LayerNorm(N_EMBED)
        self.attention = MultiHeadAttention()
        self.ff = FeedForward()

    def forward(self, x):
        context = self.attention(self.norm1(x))
        stream = x + context
        return stream + self.ff(self.norm2(stream))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_model = nn.Embedding(VOCAB_SIZE, N_EMBED)

        if POS != 'rope':
            self.positional_encoding = nn.Embedding(BLOCK_SIZE, N_EMBED)
            torch.nn.init.normal_(self.positional_encoding.weight, 0, 0.02)
        
        blocks = []
        for _ in range(N_LAYER):
            blocks.append(Block())
        self.blocks = nn.Sequential(*blocks)
        if NORM == 'rms':
            self.norm = RMSNorm(N_EMBED)
        else:
            self.norm = nn.LayerNorm(N_EMBED)        
        # weight tying: the output layer scores against the same table used to embed tokens
        self.out_scores = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.out_scores.weight = self.embedding_model.weight

        # small init so the tied table doesn't produce huge initial logits
        torch.nn.init.normal_(self.embedding_model.weight, 0, 0.02)

    def forward(self, x, y = None):
        stream = self.embedding_model(x)
        if POS != 'rope':
            positions = torch.arange(x.shape[1], device=x.device)
            stream = stream + self.positional_encoding(positions)

        stream = self.blocks(stream)
        
        logits = self.out_scores(self.norm(stream))
        
        if y is None:
            return logits
    
        logits = logits.view(x.shape[0] * x.shape[1], VOCAB_SIZE)
        y = y.view(logits.shape[0])
        loss = F.cross_entropy(logits, y)
        return logits, loss


@torch.no_grad()
def generate(model: Transformer, context: torch.Tensor, temperature = 1.0, top_k = None):
    sliced_context = context[:, -BLOCK_SIZE:]
    logits = model(sliced_context)
    scores = logits[:, -1, :]/temperature
    
    if top_k:
        last_value = torch.topk(scores, top_k)[0][:, -1:]
        scores = scores.masked_fill(scores < last_value, float('-inf'))

    scores = F.softmax(scores, dim=-1)
    index = torch.multinomial(scores, num_samples=1)
    return torch.cat((context, index), dim=1)

