from tqdm import tqdm
import torch

def get_mean_std(loader, num_channels):
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, num_channels])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, num_channels])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
