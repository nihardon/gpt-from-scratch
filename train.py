import torch
from model import Transformer, generate
from dataset import get_batch, device
import time
from tokenizer import decode


MAX_STEPS = 15000
EVAL_ITERS = 200
torch.manual_seed(1)


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
optimizer = torch.optim.AdamW(model.parameters(), 0.001)
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

    if step % 1000 == 0:
        train_loss, val_loss = estimate_loss(model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best.pt')

        print(f'Step: {step}   Train Loss: {train_loss}   Val Loss: {val_loss}')
end = time.time()
print(f'Training time: {end - start}')
print(best_val_loss)

model.eval()
context = torch.tensor([[0]], dtype=torch.long, device=device)
for _ in range(500):
    context = generate(model, context)

context = context[0].tolist()
print(decode(context))