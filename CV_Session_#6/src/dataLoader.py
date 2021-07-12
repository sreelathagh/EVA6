from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader



class Loader:
    def __init__(self,text):

        self.text = 'This class loads the data for the model'

    def transform():

        trainTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307),(0.3081)),
            transforms.RandomAffine(degrees=7,translate=(0.15,0.15), scale=(0.7, 1.3), shear=8, fill=(0,)),
        ])
        simpleTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return trainTransform, simpleTransform

    def Loader(trainTransform, simpleTransform):

        seed = 45
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(seed)
            kwargs = {'batch_size':128, 'pin_memory':True, 'num_workers':4}
        else:
            torch.manual_seed(seed)
            kwargs = {'batch_size': 64}

        
        train = MNIST(root='./data', download=True, train=True, transform= trainTransform)
        test = MNIST(root='./data', download=True, transform=simpleTransform)

        train_loader = DataLoader(train, shuffle=True, **kwargs)
        test_loader = DataLoader(test, shuffle=True, **kwargs)

        return train_loader,test_loader
