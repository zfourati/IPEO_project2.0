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

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import torch
#matplotlib.use('Agg')
#from glob import glob

print(torch.cuda.is_available())
LABEL_CLASSES = (
    "Bad data",
    "Snow and Ice",
    "Wet ice and meltwater",
    "Freshwater",
    "Sediment",
    "Bedrock",
    "Vegetation",
    )
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
                        file_name
                    ))
        else:
            files_list = sorted(os.listdir(f'data/images/train/{str(year)}'))
            for file_name in files_list:
                imgName = os.path.join(f'data/images/train/{str(year)}',file_name)
                labelName = os.path.join('data/labels/train/',file_name)
                self.data.append((
                        imgName,
                        labelName,
                        file_name
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

from collections import Counter

def plot_label_distribution(dataset):
    # Initialize a counter for label frequencies
    label_counter = Counter()

    # Iterate over the dataset
    for _, labels, _ in dataset:
        unique, counts = np.unique(labels, return_counts=True)
        label_counter.update(dict(zip(unique, counts)))

    # Map label indices to class names
    class_names = LABEL_CLASSES
    class_counts = [label_counter.get(i, 0) for i in range(len(class_names))]
    #print(sum(class_counts))
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, np.array(class_counts) / sum(class_counts), color='skyblue')
    plt.xlabel('Label Classes')
    plt.ylabel('Frequency')
    plt.title('Label Distribution in Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Label_distrib_TEST')
    plt.show()

# Example usage
dataset_train_2014 = GreenlandData(year=2014)
dataset_train_2015 = GreenlandData(year=2015)
dataset_train_2016 = GreenlandData(year=2016)
# Combine the datasets
combined_dataset = torch.utils.data.ConcatDataset([dataset_train_2014, dataset_train_2015,dataset_train_2016])
plot_label_distribution(combined_dataset)
