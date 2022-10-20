import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from random import randint

class display_data_set:
    def __init__(self):
        pass

    # Functions to display single or a batch of sample images
    def imshow(self,img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_batch(self,dataloader):
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        self.imshow(make_grid(images))  # Using Torchvision.utils make_grid function

    def show_image(self,dataloader):
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        random_num = randint(0, len(images) - 1)
        self.imshow(images[random_num])
        label = labels[random_num]
        print(f'Label: {label}, Shape: {images[random_num].shape}')


