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
from scipy.ndimage import gaussian_filter


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

    def __init__(self, transforms=False, split='train'):
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
            # Read the RGB bands (3, 2, 1)
            rgb = np.dstack([src.read(3), src.read(2), src.read(1)])
            # Normalize RGB for better visualization
            rgb = rgb.astype(float)
            if rgb.max() - rgb.min() != 0:
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        with rasterio.open(labelName) as lbl_src:
            labels = lbl_src.read(1)  # Read the first band which contains the labels

        if self.transforms == True:
            # Random horizontal flip
            if random.random() > 0.5:
                rgb = np.flip(rgb, axis=1).copy()
                labels = np.flip(labels, axis=1).copy()
            
            # Random vertical flip
            if random.random() > 0.5:
                rgb = np.flip(rgb, axis=0).copy()
                labels = np.flip(labels, axis=0).copy()

            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])  # Number of 90-degree rotations
                rgb = np.rot90(rgb, k, axes=(0, 1)).copy()
                labels = np.rot90(labels, k, axes=(0, 1)).copy()
            
            # Gaussian Blur
            if random.random() > 0.5:
                sigma = random.uniform(0.1, 2.0)  # Randomly choose a sigma value for the blur
                rgb = gaussian_filter(rgb, sigma=(sigma, sigma, 0))  # Apply Gaussian blur
            
            rgb = torch.tensor(rgb, dtype=torch.float32) # Channel-first format
            labels = torch.tensor(labels, dtype=torch.long)  # Classification labels

        return rgb, labels, fileName
            

class GreenlandData_features(Dataset):
    LABEL_CLASSES = (
        "Bad data",
        "Snow and Ice",
        "Wet ice and meltwater",
        "Freshwater",
        "Sediment",
        "Bedrock",
        "Vegetation",
    )

    def __init__(self, transforms = False, split='train'):
        self.transforms = transforms
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
            
        if self.transforms == True:
            # Random horizontal flip
            if random.random() > 0.5:
                bands = np.flip(bands, axis=1).copy()
                labels = np.flip(labels, axis=1).copy()
            
            # Random vertical flip
            if random.random() > 0.5:
                bands = np.flip(bands, axis=0).copy()
                labels = np.flip(labels, axis=0).copy()

            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])  # Number of 90-degree rotations
                bands = np.rot90(bands, k, axes=(0, 1)).copy()
                labels = np.rot90(labels, k, axes=(0, 1)).copy()
            
            # Gaussian Blur
            if random.random() > 0.5:
                sigma = random.uniform(0.1, 2.0)  # Randomly choose a sigma value for the blur
                bands = gaussian_filter(bands, sigma=(sigma, sigma, 0))  # Apply Gaussian blur

        bands = torch.tensor(bands, dtype=torch.float32).permute(2, 0, 1)  # Channel-first format
        labels = torch.tensor(labels, dtype=torch.long)  # Classification labels

        return bands, labels, fileName


class GreenlandData_Unet_features(Dataset):
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
    
def LoadData(batch_size, split, num_workers, transforms = False):
    """
    Creates and returns a reshaped DataLoader for the GreenlandData dataset.

    Args:
        batch_size (int): The number of samples per batch to load.
        split (str): The dataset split to load ('train', 'val', or 'test').
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        DataLoader: A reshaped DataLoader instance for the specified dataset split.
    """
    return ReshapeDataLoader(DataLoader(GreenlandData(transforms=transforms, split=split),
                                        batch_size=batch_size,
                                        shuffle=(split=='train'),
                                        num_workers=num_workers,))
    
class Hypercolumn(nn.Module):
    """
    A PyTorch module that implements a hypercolumn-based CNN architecture 
    for semantic segmentation or classification tasks.

    Attributes:
        block1 (nn.Sequential): First convolutional block with 32 filters.
        block2 (nn.Sequential): Second convolutional block with 64 filters.
        block3 (nn.Sequential): Third convolutional block with 128 filters.
        block4 (nn.Sequential): Fourth convolutional block with 256 filters.
        final (nn.Sequential): The final classifier block that reduces dimensionality 
            and outputs predictions for the given number of classes.
        upsample (nn.Upsample): Module for upsampling feature maps to match input dimensions.

    Methods:
        forward(x):
            Defines the forward pass of the network. Combines features from 
            different convolutional blocks to form a hypercolumn and outputs 
            predictions for each pixel.

    Args:
        input_channels (int, optional): Number of input channels in the image (default: 12).
        num_classes (int, optional): Number of output classes for classification (default: 7).
    """
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
        """
        Forward pass for the Hypercolumn model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width), 
                          where each pixel contains class probabilities or logits.
        """
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


criterion = nn.CrossEntropyLoss()
def train_epoch(data_loader, model, optimiser, device):
    """Train the model for 1 epoch

    Args:
        data_loader (DataLoader): The DataLoader instance that provides batches of training data.
        model (torch.nn.Module): The model to be trained.
        optimiser (torch.optim.Optimizer): The optimiser used for updating the model's parameters based on gradients.
        device (torch.device): The device (CPU or GPU) on which the computation is performed.

    Returns:
        model (torch.nn.Module): The model with updated parameters after the epoch
        loss_total (float): The total training loss accumulated over all batches
        oa_total (float): The overall accuracy of the model calculated over the training dataset.
    """
    # set model to training mode.
    model.train(True)
    model.to(device)

    # stats
    loss_total = 0.0
    oa_total = 0.0

    # iterate over dataset
    pBar = trange(len(data_loader))
    for idx, (data, target, _) in enumerate(data_loader):

        # put data and target onto correct device
        data, target = data.to(device), target.to(device)

        # reset gradients
        optimiser.zero_grad()

        # forward pass
        pred = model(data)

        # loss
        loss = criterion(pred, target)

        # backward pass
        loss.backward()

        # parameter update
        optimiser.step()

        # stats update
        loss_total += loss.item()
        oa_total += torch.mean((pred.argmax(1) == target).float()).item()

        # format progress bar
        pBar.set_description('Loss: {:.2f}, OA: {:.2f}'.format(
            loss_total/(idx+1),
            100 * oa_total/(idx+1)
        ))
        pBar.update(1)

    pBar.close()

    # normalize stats
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return model, loss_total, oa_total

def setup_optimiser(model, learning_rate, weight_decay):
    return SGD(
        model.parameters(),
        learning_rate,
        weight_decay
    )

def load_model(model, path_to_model, epoch='latest'):
    """
    Load saved parameters into a model.

    Args:
        model (torch.nn.Module): The model instance into which the parameters will be loaded.
        path_to_model (str): Path to the folder where the model parameters are saved.
        epoch (int or str, optional): The specific training epoch to load. Defaults to 'latest'.

    Returns:
        model (torch.nn.Module): The model with the loaded parameters.
        epoch (int): The epoch corresponding to the loaded model parameters.
    """
    modelStates = glob.glob(path_to_model + '/*.pth')
    if len(modelStates) and (epoch == 'latest' or epoch > 0):
        modelStates = [int(m.replace(path_to_model+'/','').replace('.pth', '')) for m in modelStates]
        if epoch == 'latest':
            epoch = max(modelStates)
        stateDict = torch.load(open(f'{path_to_model}/{epoch}.pth', 'rb'), map_location='cpu')
        model.load_state_dict(stateDict)
    else:
        # fresh model
        epoch = 0
    return model, epoch

def validate_epoch(data_loader, model, device):  
    """
    Validate the model for the training epoch on the validation dataset

    Args:
        data_loader (DataLoader): The DataLoader instance that provides batches of validation data.
        model (torch.nn.Module): The model to be validated.
        device (torch.device): The device (CPU or GPU) on which the computation is performed.

    Returns:
        loss_total (float): The total validation loss accumulated over all batches
        oa_total (float): The overall accuracy of the model calculated over the validation dataset.
    """

    # set model to evaluation mode
    model.train(False)
    model.to(device)

    # stats
    loss_total = 0.0
    oa_total = 0.0

    # iterate over dataset
    pBar = trange(len(data_loader))
    for idx, (data, target, _) in enumerate(data_loader):
        with torch.no_grad():
            # put data and target onto correct device
            data, target = data.to(device), target.to(device)

            # forward pass
            pred = model(data)

            # loss
            loss = criterion(pred, target)

            # stats update
            loss_total += loss.item()
            oa_total += torch.mean((pred.argmax(1) == target).float()).item()

            # format progress bar
            pBar.set_description('Loss: {:.2f}, OA: {:.2f}'.format(
                loss_total/(idx+1),
                100 * oa_total/(idx+1)
            ))
            pBar.update(1)

    pBar.close()

    # normalize stats
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return loss_total, oa_total

def save_model(model, epoch, path_to_model):
    """
    Save the model's state dictionary to a file.

    Args:
        model (torch.nn.Module): The model instance whose parameters are to be saved.
        epoch (int): The training epoch to label the saved model file.
        path_to_model (str): The directory path where the model file will be saved.

    Returns:
        None
    """
    torch.save(model.state_dict(), open(f'{path_to_model}/{epoch}.pth', 'wb'))
    
def plot_label_distribution(dataset, path_to_plot,  state = 'train'):
    """ 
    Generate a bar plot of the class distribution in the dataset and saves it.
    Args:
        dataset (DataLoader): The DataLoader instance that provides batches of size 1 of the data.
        path_to_plot (str): Path to the folder where the plot will be saved
        state (str, optional): State of the dataset used to name the plot. Defaults to 'train'.
    """
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    label_names = [
    "Bad data",
    "Snow and Ice",
    "Wet ice and meltwater",
    "Freshwater",
    "Sediment",
    "Bedrock",
    "Vegetation",
    ]
    # Initialize a counter for label frequencies
    label_counter = Counter()

    # Iterate over the dataset
    for _, labels, _ in dataset:
        unique, counts = np.unique(labels, return_counts=True)
        label_counter.update(dict(zip(unique, counts)))

    # Map label indices to class names
    class_names = label_names
    class_counts = [label_counter.get(i, 0) for i in range(len(class_names))]
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, np.array(class_counts) / sum(class_counts), color='blue')
    plt.xlabel('Label Classes')
    plt.ylabel('Frequency')
    plt.title(f'Label Distribution in {state} Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{path_to_plot}/Label_distribution_{state}.png')
    plt.show()

def visualize(dataLoader,model, path_to_plot, path_to_model, device='cuda', epoch = 'latest', numImages=5):
    """
    Generates and saves the following plots:
        - Groundtruth vs. prediction for a number of images
        - Bar plot of the Label distribution of groundtruth and the predictions of the dataset.
        - Confusion matrix

    Args:
        dataset (DataLoader): The DataLoader instance that provides batches of size 1 of the data.        
        model (torch.nn.Module): The model instance whose predictions are to be saved.
        path_to_plot (str): Path to the folder where the plot will be saved.
        path_to_model (str): Path to the folder where the model's state dictionary is stored
        device (torch.device): The device (CPU or GPU) on which the computation is performed.
        epoch (int or str, optional): The specific training epoch to load.
        numImages (int, optional): Number of images to plot Groundtruth vs. prediction. Defaults to 5.
        
    Returns:
        None
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    label_names = [
    "Bad data",
    "Snow and Ice",
    "Wet ice and meltwater",
    "Freshwater",
    "Sediment",
    "Bedrock",
    "Vegetation",
    ]
    
    file_name = path_to_plot.split('/', 1)[1] 

    model, _ = load_model(model, path_to_model, epoch)
    model = model.to(device)
    label_counter = Counter()
    pred_counter =  Counter()
    # Store true and predicted labels for confusion matrix
    all_true_labels = []
    all_pred_labels = []

    
    _, ax = plt.subplots(nrows=numImages, ncols=2,figsize = (20, 15))

    for idx, (data, labels, image_name) in enumerate(dataLoader):
        if idx == numImages:
            break
        unique_label, counts_label = np.unique(labels, return_counts=True)
        label_counter.update(dict(zip(unique_label, counts_label)))
        
        #_, ax = plt.subplots(nrows=1, ncols=2, figsize = (20, 15))
        
        labels = labels.to(device)
        
        # plot ground truth
        ax[idx,0].imshow(labels.squeeze(0).cpu().numpy(), cmap='tab20', vmin=0, vmax=len(label_names) - 1)
        ax[idx,0].axis('off')
        ax[idx,0].set_title(f'Ground Truth: {image_name[0]}')


        with torch.no_grad():
            pred = model(data.to(device))

            # get the label (i.e., the maximum position for each pixel along the class dimension)
            yhat = torch.argmax(pred, dim=1)
            
            unique_pred, counts_pred = np.unique(yhat.cpu().numpy(), return_counts=True)
            pred_counter.update(dict(zip(unique_pred, counts_pred)))

            # Append to confusion matrix lists
            all_true_labels.extend(labels.cpu().numpy().flatten())
            all_pred_labels.extend(yhat.cpu().numpy().flatten())
            
            # plot model predictions
            ax[idx,1].imshow(yhat.squeeze(0).cpu().numpy(), cmap='tab20', vmin=0, vmax=len(label_names) - 1)
            ax[idx,1].axis('off')
            ax[idx,1].set_title(f'Prediction: {image_name[0]}', fontsize=16)

    plt.tight_layout()
    plt.savefig(f'{path_to_plot}/GroundtruthVspred_{file_name}.png')


    class_counts_labels = [label_counter.get(i, 0) for i in range(len(label_names))]
    class_counts_pred = [pred_counter.get(i, 0) for i in range(len(label_names))]

    SMALL_SIZE = 20
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 30

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.35  # width of the bars
    x = np.arange(len(label_names))  # the label positions

    # Plot the bars for ground truth (blue) and predicted classes (green)
    ax.bar(x - width / 2, np.array(class_counts_labels)/sum(class_counts_labels), width, label='Ground Truth', color='blue', alpha=0.7)
    ax.bar(x + width / 2, np.array(class_counts_pred)/sum(class_counts_pred), width, label='Predicted', color='green', alpha=0.7)

    # Set the labels and titles
    ax.set_title(f'Class Frequency Comparison: Ground Truth vs. Predicted ({file_name}) ')
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{path_to_plot}/Label_distribution_test_{file_name}.png')
    
    #Plot confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=np.arange(len(label_names)))

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca())


    plt.title(f'Confusion Matrix ({file_name})')
    plt.tight_layout()
    plt.savefig(f'{path_to_plot}/Confusion_Matrix_{file_name}.png')

