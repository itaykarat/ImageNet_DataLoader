import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets
from CODE import constants
from torchvision import transforms as T


class data_loader:
    def __init__(self):
        pass

    # Setup function to create dataloaders for image datasets
    def generate_dataloader(self,data, name, transform):

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if data is None:
            return None

        # Read image files to pytorch dataset using ImageFolder, a generic data
        # loader where images are in format root/label/filename
        # See https://pytorch.org/vision/stable/datasets.html
        if transform is None:
            dataset = datasets.ImageFolder(data, transform=T.ToTensor())
        else:
            dataset = datasets.ImageFolder(data, transform=transform)

        # Set options for device
        if use_cuda:
            kwargs = {"pin_memory": True, "num_workers": 1}
        else:
            kwargs = {}

        # Wrap image dataset (defined above) in dataloader
        dataloader = DataLoader(dataset, batch_size=constants.batch_size,
                                shuffle=(name == "train"),
                                **kwargs)

        return dataloader