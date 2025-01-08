"""
Baseline Project
We combine the datasets for the 3 years and split them into 80% training data and 20% validation.
Model used: Hypercolumn
"""

import torch
import rasterio
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, random_split, DataLoader
from PIL import Image
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import Counter
from segmentation_models_pytorch import Unet

print(torch.cuda.is_available())
seed = 323444           # the seed value used to initialise the random number generator of PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

path_to_model = 'cnn_states/PretrainedUnet'
os.makedirs(path_to_model, exist_ok=True)

path_to_plot = 'Plots/PretrainedUnet'
os.makedirs(path_to_plot, exist_ok=True)

class GreenlandData(Dataset):
    LABEL_CLASSES = (
    "Bad data",
    "Snow and Ice",
    "Wet ice and meltwater",
    "Freshwater",
    "Sediment",
    "Bedrock",
    "Vegetation",
    )

    def __init__(self, year=2014, transforms=None):
        self.transforms = transforms

    # prepare data
        self.data = []  # list of tuples of (image path, label path, name)
        if year == 2023:
            files_list = sorted(os.listdir('data/images/test/2023'))
            for file_name in files_list:
                imgName = os.path.join('data/images/test/2023/',file_name)
                labelName = os.path.join('data/labels/test/',file_name)
                self.data.append((
                        imgName,
                        labelName,
                        file_name.replace(".tif", "")
                    ))
        else:
            files_list = sorted(os.listdir(f'data/images/train/{str(year)}'))
            for file_name in files_list:
                imgName = os.path.join(f'data/images/train/{str(year)}',file_name)
                labelName = os.path.join('data/labels/train/',file_name)
                self.data.append((
                        imgName,
                        labelName,
                        file_name.replace(".tif", "")
                    ))


    def __len__(self):
            return len(self.data)


    def __getitem__(self, x):
        imgName, labelName, fileName = self.data[x]
        with rasterio.open(imgName) as src:
            # Read the RGB bands (3, 2, 1)
            rgb = np.dstack([src.read(3), src.read(2), src.read(1)])
            # Normalize RGB for better visualization
            rgb = rgb.astype(float)
            if rgb.max() - rgb.min() != 0:
                #print('min:',rgb.min())
                #print('max:',rgb.max())
                #print(fileName)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        if self.transforms is not None:
            rgb = self.transforms(rgb)

        with rasterio.open(labelName) as lbl_src:
            labels = lbl_src.read(1)  # Read the first band which contains the labels
        return rgb, labels, fileName

class ReshapeDataLoader:
    def __init__(self, dataloader):
        """
        A wrapper to reshape data from a PyTorch DataLoader.
        Args:
            dataloader (DataLoader): The original DataLoader.
        """
        self.dataloader = dataloader

    def __iter__(self):
        """
        Iterate over the DataLoader and reshape the data.
        """
        for data, target, img_name in self.dataloader:
            # Ensure data is float and properly reshaped
            data = data.float()
            if data.ndim == 4 and data.shape[-1] != data.shape[1]:
                data = data.permute(0, 3, 1, 2)

            # Ensure target is long and properly reshaped
            target = target.long()
            if target.ndim == 4 and target.shape[-1] != target.shape[1]:
                target = target.permute(0, 3, 1, 2)

            yield data, target, img_name

    def __len__(self):
        """
        Return the length of the DataLoader.
        """
        return len(self.dataloader)



dataset = GreenlandData()
print(f"dataset of length {len(dataset)}")
# Labels as categories
label_names = [
    "Bad data",
    "Snow and Ice",
    "Wet ice and meltwater",
    "Freshwater",
    "Sediment",
    "Bedrock",
    "Vegetation",
    ]

from ipywidgets import interact
#@interact(idx=range(len(dataset)))
#def plot_sample(idx=0):
for idx in range(3):
    img,label, name = dataset[idx]
    plt.figure()
    plt.axis('off')
    plt.title(name)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='tab20', vmin=0, vmax=len(label_names) - 1)
    plt.axis('off')

    plt.tight_layout()
plt.show()

