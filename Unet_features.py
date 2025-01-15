"""

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
from tqdm.notebook import trange
import random
from torch.optim import SGD
import glob
from segmentation_models_pytorch import Unet
import segmentation_models_pytorch as smp
import Function_lib as lib

print('GPU available: ',torch.cuda.is_available())
seed = 323444           # the seed value used to initialise the random number generator of PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

path_to_model = 'cnn_states/pretrainedUnet_features'
os.makedirs(path_to_model, exist_ok=True)

path_to_plot = 'Plots/pretrainedUnet_features'
os.makedirs(path_to_plot, exist_ok=True)

# define hyperparameters
device = 'cuda'
start_epoch = 0 # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
batch_size = 30
learning_rate = 0.1
weight_decay = 0.001
num_epochs = 10
validation_split_ratio = 0.2

label_names = [
    "Bad data",
    "Snow and Ice",
    "Wet ice and meltwater",
    "Freshwater",
    "Sediment",
    "Bedrock",
    "Vegetation",
    ]

model = Unet(
   encoder_name="resnet34",        # Use a ResNet-34 encoder
   encoder_weights="imagenet",    # Pre-trained on ImageNet
   in_channels=3,                 # Number of input channels (e.g., RGB images)
    classes=7                      # Number of output classes
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

    def __init__(self, split='train', transforms=None):
        self.transforms = transforms

    # prepare data
        self.data = []  # list of tuples of (image path, label path, name)
        if split == 'test':
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
            seed = 323444 
            random.seed(seed)
            all_image_name = os.listdir('data/images/train/2014')
            random.shuffle(all_image_name)
            validation_split_ratio = 0.2
            validation_size = int(validation_split_ratio * len(all_image_name))
            # Split the list
            val_list = all_image_name[:validation_size]
            train_list = all_image_name[validation_size:]
            if split == 'train':
                for file_name in train_list:
                    for year in ['2014','2015','2016']:
                        imgName = os.path.join(f'data/images/train/{year}',file_name)
                        labelName = os.path.join('data/labels/train/',file_name)
                        self.data.append((
                                imgName,
                                labelName,
                                file_name.replace(".tif", "")
                            ))
            elif split == 'val':
                for file_name in val_list:
                    for year in ['2014','2015','2016']:
                        imgName = os.path.join(f'data/images/train/{year}',file_name)
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
             # Calculate indices
                red = src.read(3)  # Band 4: Red
                nir = src.read(4)  # Band 5: Near Infrared
                green = src.read(2)  # Band 3: Green
                swir1 = src.read(5)  # Band 6: Shortwave Infrared 1

                ndvi = (nir - red) / (nir + red + 1e-6)
                ndwi = (green - nir) / (green + nir + 1e-6)
                ndsi = (green - swir1) / (green + swir1 + 1e-6)

                bands = np.dstack([ndvi, ndwi, ndsi])

        if self.transforms is not None:
            bands = self.transforms(bands)

        with rasterio.open(labelName) as lbl_src:
            labels = lbl_src.read(1)  # Read the first band which contains the labels
        return bands, labels, fileName


criterion = nn.CrossEntropyLoss()
lib.ReshapeDataLoader(DataLoader(GreenlandData(split='train'),
                                        batch_size=batch_size,
                                        shuffle='True',
                                        num_workers=1))

dl_train = lib.ReshapeDataLoader(DataLoader(GreenlandData(split='train'), batch_size=batch_size, shuffle='True', num_workers=1))
dl_val = lib.ReshapeDataLoader(DataLoader(GreenlandData(split='val'), batch_size=batch_size, shuffle='False', num_workers=1))

# load model
model, epoch = lib.load_model(model, path_to_model, epoch=start_epoch)
optim = lib.setup_optimiser(model, learning_rate, weight_decay)


# do epochs
while epoch < num_epochs:
    print(epoch)

    # training
    model, loss_train, oa_train = lib.train_epoch(dl_train, model, optim, device)

    # validation
    loss_val, oa_val = lib.validate_epoch(dl_val, model, device)

    # print stats
    print('[Ep. {}/{}] Loss train: {:.2f}, val: {:.2f}; OA train: {:.2f}, val: {:.2f}'.format(
        epoch+1, num_epochs,
        loss_train, loss_val,
        100*oa_train, 100*oa_val
    ))

    # save model
    epoch += 1
    lib.save_model(model, epoch, path_to_model)
    
    
#Testing
dl_test = lib.ReshapeDataLoader(DataLoader(GreenlandData(split='test'), batch_size=batch_size, shuffle='False', num_workers=1))
loss_test, oa_test = lib.validate_epoch(dl_test, model, device)
print('Testing:  Loss: {:.2f}  OA: {:.2f}'.format(loss_test, 100*oa_test))

#Visualize predictions and label class distribution
dl_test_single = lib.ReshapeDataLoader(DataLoader(GreenlandData(split='test'), batch_size=1, shuffle='False', num_workers=1))          
lib.visualize(dl_test_single, model, path_to_plot, path_to_model)

dl_train_single = lib.ReshapeDataLoader(DataLoader(GreenlandData(split='train'), batch_size=1, shuffle='False', num_workers=1))
dl_val_single = lib.ReshapeDataLoader(DataLoader(GreenlandData(split='train'), batch_size=1, shuffle='False', num_workers=1))
lib.plot_label_distribution(dl_train_single, path_to_plot)
lib.plot_label_distribution(dl_train_single,path_to_plot,  state = 'val')

