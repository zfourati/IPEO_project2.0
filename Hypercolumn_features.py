"""
This script processes Greenland imagery to combine temporal statistics for landcover classification. 
It computes features such as NDVI, NDWI, and NDSI, along with their mean and standard deviation over multiple years 
(2014, 2015, 2016 for training/validation and 2023 for testing). These temporal features are integrated into 
a hypercolumn representation for deep learning-based classification using PyTorch. 
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
import Function_lib as lib

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

class GreenlandData_Features(Dataset):
    """
    PyTorch Dataset for loading Greenland imagery and labels with temporal features.

    Attributes:
        LABEL_CLASSES (tuple): Class labels for the dataset.

    Methods:
        __init__(split): Initializes the dataset based on the specified split ('train', 'val', 'test').
        __len__(): Returns the number of samples in the dataset.
        __getitem__(x): Retrieves the features, labels, and filename for a given index.
    """
    
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
        """
        Initializes the dataset and prepares data paths based on the split.

        Args:
            split (str): The dataset split ('train', 'val', or 'test').
        """
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
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, x):
        """
        Retrieves the features, labels, and filename for a given index.

        Args:
            x (int): Index of the sample.

        Returns:
            tuple: Features (torch.Tensor), labels (torch.Tensor), and filename (str).
        """
        imgPaths, labelName, fileName = self.data[x]

        # Read and process temporal images (for 2014, 2015, 2016 or 2023 for test)
        temporal_bands = [] #List of RGB bands for (2014, 2015 and 2016) or 2023 for test
        temporal_ndvi = []  #List of NDVI values for (2014, 2015 and 2016) or 2023 for test
        temporal_ndwi = []  #List of NDWI values for (2014, 2015 and 2016) or 2023 for test
        temporal_ndsi = []  #List of NDSI values for (2014, 2015 and 2016) or 2023 for test

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
        rgb_mean = np.mean(temporal_bands, axis=0)  #Mean RGB bands across (2014, 2015 and 2016) or 2023 for test
        rgb_std = np.std(temporal_bands, axis=0)    #Standard deviation of RGB bands across (2014, 2015 and 2016) or 2023 for test
        ndvi_mean = np.mean(temporal_ndvi, axis=0)  #Mean NDVI values across (2014, 2015 and 2016) or 2023 for test
        ndvi_std = np.std(temporal_ndvi, axis=0)    #Standard deviation of NDVI values across (2014, 2015 and 2016) or 2023 for test
        ndwi_mean = np.mean(temporal_ndwi, axis=0)  #Mean NDWI values across (2014, 2015 and 2016) or 2023 for test
        ndwi_std = np.std(temporal_ndwi, axis=0)    #Standard deviation of NDWI values across (2014, 2015 and 2016) or 2023 for test
        ndsi_mean = np.mean(temporal_ndsi, axis=0)  #Mean NDSI values across (2014, 2015 and 2016) or 2023 for test
        ndsi_std = np.std(temporal_ndsi, axis=0)    #Standard deviation of NDSI values across (2014, 2015 and 2016) or 2023 for test

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
    
criterion = nn.CrossEntropyLoss()

#Load training and validation data
dl_train = DataLoader(GreenlandData_Features(split='train'), batch_size=batch_size, num_workers=1)
dl_val = DataLoader(GreenlandData_Features(split='val'), batch_size=batch_size, num_workers=1)

# load model
model, epoch = lib.load_model(lib.Hypercolumn(input_channels=12), path_to_model, epoch=start_epoch)
optim = lib.setup_optimiser(model, learning_rate, weight_decay)

# do epochs
while epoch < num_epochs:

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
dl_test = DataLoader(GreenlandData_Features(split='test'),batch_size= batch_size, num_workers=1)
loss_test, oa_test = lib.validate_epoch(dl_test, model, device)
print('Testing:  Loss: {:.2f}  OA: {:.2f}'.format(loss_test, 100*oa_test))

#Visualize predictions and label class distribution
dl_test_single = DataLoader(GreenlandData_Features(split='test'),batch_size= 1, num_workers=1)         
lib.visualize(dl_test_single,model, path_to_plot, path_to_model)

dl_train_single = DataLoader(GreenlandData_Features(split='train'),batch_size= batch_size, num_workers=1)
dl_val_single = DataLoader(GreenlandData_Features(split='val'),batch_size= batch_size, num_workers=1)
lib.plot_label_distribution(dl_train_single, path_to_plot)
lib.plot_label_distribution(dl_train_single,path_to_plot,  state = 'val')