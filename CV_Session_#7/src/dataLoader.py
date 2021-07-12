from torchvision.datasets import MNIST, CIFAR10
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
from torch.utils.data import DataLoader
import torchvision
import numpy as np


DATA_MEAN = (0.4914, 0.4822, 0.4465)
DATA_STD = (0.247, 0.2435, 0.2616)

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


class Loader:
    def __init__(self, batch_size):

        self.text = 'This class loads the data for the model'
        self.batch_size=batch_size

    def transform(self):

        trainTransform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5),
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=DATA_MEAN, mask_fill_value = None),
            A.Normalize(DATA_MEAN, DATA_STD),
            ToTensorV2(),
        ])
        simpleTransform = A.Compose([
            A.Normalize(DATA_MEAN, DATA_STD),
            ToTensorV2(),
        ])
        return Transforms(trainTransform), Transforms(simpleTransform)

    def Loader(self,trainTransform, simpleTransform, cuda: bool=True):
        seed = 42
        if cuda:
            torch.cuda.manual_seed(seed)
            kwargs = {'batch_size': self.batch_size, 'pin_memory': True, 'num_workers': 4}
        else:
            torch.manual_seed(seed)
            kwargs = {'batch_size': self.batch_size}

        train = CIFAR10(root='./data', train=True,
                        download=True, transform=trainTransform)
        test = CIFAR10(root='./data', download=True, transform=simpleTransform)

        train_loader = DataLoader(train, shuffle=True, **kwargs)
        test_loader = DataLoader(test, shuffle=True, **kwargs)

        return train_loader, test_loader

