from tokenizer import TEXT, encode
import torch

data = torch.tensor(encode(TEXT), dtype=torch.long)

split_i = (len(data) * 9 // 10)

train_data = data[:split_i]
val_data = data[split_i:]

block_size = 8
batch_size = 4

def get_batch(dt: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_x = []
    batch_y = []
    d = train_data
    if dt == 'val':
        d = val_data


    for _ in range(batch_size):
        i = torch.randint(len(d)-block_size-1, (1,))
        batch_x.append(d[i:i+block_size])
        batch_y.append(d[i+1:i+block_size+1])

    return torch.stack(batch_x), torch.stack(batch_y)

print(get_batch('train')[0])