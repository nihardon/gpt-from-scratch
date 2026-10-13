from tokenizer import TEXT, encode
import torch

data = torch.tensor(encode(TEXT), dtype=torch.long)

split_i = (len(data) * 9 // 10)

train_data = data[:split_i]
val_data = data[split_i:]

BLOCK_SIZE = 8
BATCH_SIZE = 4

def get_batch(dt: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_x = []
    batch_y = []
    d = train_data
    if dt == 'val':
        d = val_data


    for _ in range(BATCH_SIZE):
        i = torch.randint(len(d)-BLOCK_SIZE-1, (1,))
        batch_x.append(d[i:i+BLOCK_SIZE])
        batch_y.append(d[i+1:i+BLOCK_SIZE+1])

    return torch.stack(batch_x), torch.stack(batch_y)

