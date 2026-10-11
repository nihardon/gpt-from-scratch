with open('data/input.txt', 'r') as file:
    TEXT = file.read()

vocab = set(TEXT)
vocab = list(vocab)
vocab.sort()


stoi = {}
itos = {}

for i, c in enumerate(vocab):
    stoi[c] = i
    itos[i] = c

def encode(s: str) -> list[int]:
    l = []
    for c in s:
        l.append(stoi[c])
    return l

def decode(l: list[int]) -> str:
    s = []
    for i in l:
        s.append(itos[i])
    return ''.join(s)

