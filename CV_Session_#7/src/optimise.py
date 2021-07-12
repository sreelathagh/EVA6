from torch import optim
from torch.optim import lr_scheduler
from .train import train
from .test import test



def learner(model, train_loader, test_loader, epochs, device):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    optimizer = optim.Adam(model.parameters(), lr=0.03)
    epoch = epochs

    for epoch in range(1, epoch + 1):
        print(f'Epoch {epoch}:')

        train(model, device, train_loader, optimizer,
                train_acc, train_losses)

        test(model, device, test_loader, test_acc, test_losses)

    return (train_acc, train_losses, test_acc, test_losses), model
