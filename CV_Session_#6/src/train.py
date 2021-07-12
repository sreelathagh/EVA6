import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train(model, device, train_loader, optimizer, train_losses, train_acc, lambda_l1):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):

    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    y_pred = model(data)

    loss = F.nll_loss(y_pred, target)

    # l1 regularization
    if lambda_l1 > 0:
      l1 = 0
      for p in model.parameters():
        l1 += p.abs().sum()
      loss += lambda_l1 * l1

    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    pred = y_pred.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(
        desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
