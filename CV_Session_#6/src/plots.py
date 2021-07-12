import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class Plots:
    def __init__(self):
        pass

    def sampleVisual(dataset):
        batch = next(iter(dataset))
        images, labels = batch
        batch_grid = make_grid(images)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        return plt.imshow(batch_grid[0].squeeze(), cmap='gray_r')
