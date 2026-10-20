import torch
from model import Transformer, generate
from dataset import get_batch, device
import time
from tokenizer import decode, EOT
from config import MAX_STEPS, EVAL_ITERS, LR, SEED, CHECKPOINT, EVAL_INTERVAL

torch.manual_seed(SEED)

@torch.no_grad()
def estimate_loss(model: Transformer):
    model.eval()
    
    train_loss = 0
    val_loss = 0
    for _ in range(EVAL_ITERS):
        x, y = get_batch('train')
        train_loss += model(x, y)[1]

        x, y = get_batch('val')
        val_loss += model(x, y)[1]

    model.train()

    return train_loss/EVAL_ITERS, val_loss/EVAL_ITERS

model = Transformer()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), LR)
params = 0
for layer in model.parameters():
     params += layer.numel()
print(params)

best_val_loss = float('inf')

start = time.time()
for step in range(MAX_STEPS):
    x, y = get_batch('train')

    loss = model(x, y)[1]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % EVAL_INTERVAL == 0:
        train_loss, val_loss = estimate_loss(model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)

        print(f'Step: {step}   Train Loss: {train_loss}   Val Loss: {val_loss}')
end = time.time()
print(f'Training time: {end - start}')
print(best_val_loss)

model.eval()
context = torch.tensor([[EOT]], dtype=torch.long, device=device)
for _ in range(500):
    context = generate(model, context)

context = context[0].tolist()
print(decode(context))