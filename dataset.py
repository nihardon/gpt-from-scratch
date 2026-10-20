import torch
import numpy as np

BLOCK_SIZE = 256
BATCH_SIZE = 32

device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

train_data = np.memmap('data/train.bin', np.uint16, 'r')
val_data = np.memmap('data/val.bin', np.uint16, 'r')

def get_batch(dt: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_x = []
    batch_y = []
    d = train_data
    if dt == 'val':
        d = val_data

    for _ in range(BATCH_SIZE):
        i = torch.randint(len(d)-BLOCK_SIZE-1, (1,)).item()
        
        x_row = d[i:i+BLOCK_SIZE].astype(np.int64)
        batch_x.append(torch.from_numpy(x_row))
        
        y_row = d[i+1:i+BLOCK_SIZE+1].astype(np.int64)
        batch_y.append(torch.from_numpy(y_row))

    x, y = torch.stack(batch_x), torch.stack(batch_y)
    x = x.to(device)
    y = y.to(device)
    return x, y

