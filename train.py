import math
import time
import torch
from model import Transformer
from dataset import get_batch
from config import (MAX_STEPS, EVAL_ITERS, EVAL_INTERVAL, MAX_LR, MIN_LR, WARMUP_STEPS,
                    SEED, CHECKPOINT, device)

torch.manual_seed(SEED)


@torch.no_grad()
def estimate_loss(model: Transformer):
    model.eval()

    train_loss = 0
    val_loss = 0
    for _ in range(EVAL_ITERS):
        x, y = get_batch('train')
        train_loss += model(x, y)[1].item()

        x, y = get_batch('val')
        val_loss += model(x, y)[1].item()

    model.train()

    return train_loss/EVAL_ITERS, val_loss/EVAL_ITERS


def learning_rate(step: int) -> float:
    # linear warmup, then cosine decay from MAX_LR to MIN_LR
    if step < WARMUP_STEPS:
        return MAX_LR * (step+1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    p = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * p))


if __name__ == '__main__':
    model = Transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), MAX_LR)
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')

    best_val_loss = float('inf')
    start = time.time()
    for step in range(MAX_STEPS):
        x, y = get_batch('train')

        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group['lr'] = lr

        loss = model(x, y)[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # (step+1) so the final step is always evaluated
        if (step+1) % EVAL_INTERVAL == 0:
            train_loss, val_loss = estimate_loss(model)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT)

            print(f'step {step+1:6d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e}')

    print(f'training time: {time.time() - start:.0f}s | best val: {best_val_loss:.4f} -> {CHECKPOINT}')
