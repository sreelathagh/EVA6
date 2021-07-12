from torch import optim
from torch.optim import lr_scheduler
from .train import train
from .test import test


def learner(model, train_loader, test_loader, lambda_l1, epochs, device):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    optimizer = optim.SGD(model.parameters(), lr=0.015,
                          momentum=0.7)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)
    epoch = epochs

    for epoch in range(1, epoch + 1):
        print(f'Epoch {epoch}:')

        train(model, device, train_loader, optimizer,
                train_acc, train_losses, lambda_l1)
        scheduler.step()
        test(model, device, test_loader, test_acc, test_losses)

    return (train_acc, train_losses, test_acc, test_losses), model

