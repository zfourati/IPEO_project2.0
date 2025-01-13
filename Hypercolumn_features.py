"""
Add features to data and use on hypercolumn 
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Function_lib import *

print('GPU available: ',torch.cuda.is_available())
seed = 323444           # the seed value used to initialise the random number generator of PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

path_to_model = 'cnn_states/HypercolumnFeatures'
os.makedirs(path_to_model, exist_ok=True)

path_to_plot = 'Plots/HypercolumnFeatures'
os.makedirs(path_to_plot, exist_ok=True)

# define hyperparameters
device = 'cuda'
start_epoch = 'latest' # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
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

    def __init__(self, split='train'):
        # Prepare data
        self.data = []  # List of tuples of (image paths for years, label path, name)
        if split == 'test':
            files_list = sorted(os.listdir('data/images/test/2023'))
            for file_name in files_list:
                imgName = os.path.join('data/images/test/2023/', file_name)
                labelName = os.path.join('data/labels/test/', file_name)
                self.data.append((
                    [imgName],  # Only 2023 image for test
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
                    imgPaths = [os.path.join(f'data/images/train/{year}', file_name) for year in ['2014', '2015', '2016']]
                    labelName = os.path.join('data/labels/train/', file_name)
                    self.data.append((imgPaths, labelName, file_name.replace(".tif", "")))
            elif split == 'val':
                for file_name in val_list:
                    imgPaths = [os.path.join(f'data/images/train/{year}', file_name) for year in ['2014', '2015', '2016']]
                    labelName = os.path.join('data/labels/train/', file_name)
                    self.data.append((imgPaths, labelName, file_name.replace(".tif", "")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        imgPaths, labelName, fileName = self.data[x]

        # Read and process temporal images (for 2014, 2015, 2016 or 2023 for test)
        temporal_bands = []
        temporal_ndvi = []
        temporal_ndwi = []
        temporal_ndsi = []

        for imgPath in imgPaths:
            with rasterio.open(imgPath) as src:
                # Read RGB and other bands
                rgb = np.dstack([src.read(3), src.read(2), src.read(1)])
                rgb = rgb.astype(float)
                if rgb.max() - rgb.min() != 0:
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

                # Calculate indices
                red = src.read(3)  # Band 4: Red
                nir = src.read(4)  # Band 5: Near Infrared
                green = src.read(2)  # Band 3: Green
                swir1 = src.read(5)  # Band 6: Shortwave Infrared 1

                ndvi = (nir - red) / (nir + red + 1e-6)
                ndwi = (green - nir) / (green + nir + 1e-6)
                ndsi = (green - swir1) / (green + swir1 + 1e-6)

                temporal_bands.append(rgb)
                temporal_ndvi.append(ndvi)
                temporal_ndwi.append(ndwi)
                temporal_ndsi.append(ndsi)

        # Compute temporal features
        temporal_bands = np.stack(temporal_bands, axis=0)  # [Years, Height, Width, Channels]
        temporal_ndvi = np.stack(temporal_ndvi, axis=0)  # [Years, Height, Width]
        temporal_ndwi = np.stack(temporal_ndwi, axis=0)
        temporal_ndsi = np.stack(temporal_ndsi, axis=0)

        # Temporal statistics
        rgb_mean = np.mean(temporal_bands, axis=0)
        rgb_std = np.std(temporal_bands, axis=0)
        ndvi_mean = np.mean(temporal_ndvi, axis=0)
        ndvi_std = np.std(temporal_ndvi, axis=0)
        ndwi_mean = np.mean(temporal_ndwi, axis=0)
        ndwi_std = np.std(temporal_ndwi, axis=0)
        ndsi_mean = np.mean(temporal_ndsi, axis=0)
        ndsi_std = np.std(temporal_ndsi, axis=0)

        # Combine features into a hypercolumn
        bands = np.dstack([
            rgb_mean, rgb_std, ndvi_mean, ndvi_std, ndwi_mean, ndwi_std, ndsi_mean, ndsi_std
        ])  # [Height, Width, Channels]

        # Read labels
        with rasterio.open(labelName) as lbl_src:
            labels = lbl_src.read(1)  # Read the first band which contains the labels
            
        bands = torch.tensor(bands, dtype=torch.float32).permute(2, 0, 1)  # Channel-first format
        labels = torch.tensor(labels, dtype=torch.long)  # Classification labels

        return bands, labels, fileName


class Hypercolumn(nn.Module):
    def __init__(self, input_channels=12, num_classes=7):
        super(Hypercolumn, self).__init__()

        # Initial convolutional blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(input_channels + 32 + 64 + 128 + 256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

        # Upsampling module
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        upsample = nn.Upsample(size=(x.size(2), x.size(3)))
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        hypercol = torch.cat(
            (x, upsample(x1), upsample(x2), upsample(x3), upsample(x4)),
            dim=1
        )
        return self.final(hypercol)
"""
batch_size=3
dataset_train = GreenlandData(split='train')
dataloader_train = DataLoader(GreenlandData(split='train'), batch_size=batch_size, num_workers=2)
model = Hypercolumn()
data, _ , __= iter(dataloader_train).__next__()

pred = model(data)


assert pred.size(1) == len(dataset_train.LABEL_CLASSES), f'ERROR: invalid number of model output channels (should be # classes {len(dataset_train.LABEL_CLASSES)}, got {pred.size(1)})'
assert pred.size(2) == data.size(2), f'ERROR: invalid spatial height of model output (should be {data.size(2)}, got {pred.size(2)})'
assert pred.size(3) == data.size(3), f'ERROR: invalid spatial width of model output (should be {data.size(3)}, got {pred.size(3)})'

"""
    
criterion = nn.CrossEntropyLoss()

dl_train = DataLoader(GreenlandData(split='train'), batch_size=batch_size, num_workers=1)
dl_val = DataLoader(GreenlandData(split='val'), batch_size=batch_size, num_workers=1)

# load model
model, epoch = load_model(Hypercolumn(), path_to_model, epoch=start_epoch)
optim = setup_optimiser(model, learning_rate, weight_decay)

# do epochs
while epoch < num_epochs:

    # training
    model, loss_train, oa_train = train_epoch(dl_train, model, optim, device)

    # validation
    loss_val, oa_val = validate_epoch(dl_val, model, device)

    # print stats
    print('[Ep. {}/{}] Loss train: {:.2f}, val: {:.2f}; OA train: {:.2f}, val: {:.2f}'.format(
        epoch+1, num_epochs,
        loss_train, loss_val,
        100*oa_train, 100*oa_val
    ))

    # save model
    epoch += 1
    save_model(model, epoch, path_to_model)
    
    
#Testing
dl_test = DataLoader(GreenlandData(split='test'),batch_size= batch_size, num_workers=1)
loss_test, oa_test = validate_epoch(dl_test, model, device)
print('Testing:  Loss: {:.2f}  OA: {:.2f}'.format(loss_test, 100*oa_test))

#Visualize predictions and label class distribution
dl_test_single = DataLoader(GreenlandData(split='test'),batch_size= 1, num_workers=1)         
visualize(dl_test_single,model, path_to_plot, path_to_model)

dl_train_single = DataLoader(GreenlandData(split='train'),batch_size= batch_size, num_workers=1)
dl_val_single = DataLoader(GreenlandData(split='val'),batch_size= batch_size, num_workers=1)
plot_label_distribution(dl_train_single, path_to_plot)
plot_label_distribution(dl_train_single,path_to_plot,  state = 'val')
