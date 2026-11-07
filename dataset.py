import torch
import numpy as np
from config import BLOCK_SIZE, BATCH_SIZE, TRAIN, VAL, device

train_data = np.memmap(TRAIN, np.uint16, 'r')
val_data = np.memmap(VAL, np.uint16, 'r')


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = val_data if split == 'val' else train_data

    batch_x = []
    batch_y = []
    for _ in range(BATCH_SIZE):
        i = torch.randint(len(d) - BLOCK_SIZE - 1, (1,)).item()
        batch_x.append(torch.from_numpy(d[i:i+BLOCK_SIZE].astype(np.int64)))
        batch_y.append(torch.from_numpy(d[i+1:i+BLOCK_SIZE+1].astype(np.int64)))

    x, y = torch.stack(batch_x), torch.stack(batch_y)
    return x.to(device), y.to(device)
