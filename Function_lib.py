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


# Random crop
def random_crop(image, label, crop_height, crop_width):
    """
            Perform a random crop on an image and its corresponding label.

                Args:
                    image (np.ndarray): The input image (H, W, C) or (H, W) for grayscale.
                    label (np.ndarray): The corresponding label mask (H, W).
                    crop_height (int): Desired height of the crop.
                    crop_width (int): Desired width of the crop.

                Returns:
                    tuple: Cropped image and label.
                """
    # Get dimensions of the image
    img_height, img_width = image.shape[:2]

                # Ensure the crop size is smaller than the image dimensions
    if crop_height > img_height or crop_width > img_width:
        raise ValueError("Crop size must be smaller than the image dimensions.")

                # Randomly select the top and left coordinates
    top = np.random.randint(0, img_height - crop_height + 1)
    left = np.random.randint(0, img_width - crop_width + 1)

    # Crop the image
    cropped_image = image[top:top + crop_height, left:left + crop_width]

    # Crop the label
    cropped_label = label[top:top + crop_height, left:left + crop_width]

    return cropped_image, cropped_label
            

class GreenlandData_transforms(Dataset):
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

        # prepare data
        self.split = split
        self.data = []  # list of tuples of (image path, label path, name)
        if self.split == 'test':
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
            if self.split == 'train':
                for file_name in train_list:
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
            bands = np.dstack([src.read(3), src.read(2), src.read(1)])
            
            # Normalize each band for better visualization (optional, depending on usage)
            bands = bands.astype(float)
            for i in range(bands.shape[-1]):
                band = bands[..., i]
                if band.max() - band.min() != 0:
                    bands[..., i] = (band - band.min()) / (band.max() - band.min())
            
            # Calculate NDVI (Normalized Difference Vegetation Index)
            red = src.read(3)  # Band 4: Red
            nir = src.read(4)  # Band 5: Near Infrared
            ndvi = (nir - red) / (nir + red + 1e-6)  # Avoid division by zero
            
            # Calculate NDWI (Normalized Difference Water Index)
            green = src.read(2)  # Band 3: Green
            ndwi = (green - nir) / (green + nir + 1e-6)
            
            # Calculate NDSI (Normalized Difference Snow Index)
            swir1 = src.read(5)  # Band 6: Shortwave Infrared 1
            ndsi = (green - swir1) / (green + swir1 + 1e-6)
            
            # Stack the additional indices with the original bands
            all_bands = np.dstack([bands, ndvi, ndwi, ndsi])
        
        with rasterio.open(labelName) as lbl_src:
            labels = lbl_src.read(1)  # Read the first band which contains the labels
        
        if self.split == 'train':
            # Random horizontal flip
            if random.random() > 0.5:
                all_bands = np.flip(all_bands, axis=1)
                labels = np.flip(labels, axis=1)
            
            # Random vertical flip
            if random.random() > 0.5:
                all_bands = np.flip(all_bands, axis=0)
                labels = np.flip(labels, axis=0)

            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])  # Number of 90-degree rotations
                all_bands = np.rot90(all_bands, k, axes=(0, 1))
                labels = np.rot90(labels, k, axes=(0, 1))
                
            #if random.random() > 0.5:
             #   crop_width = np.random.randint(50, 110)
              #  crop_height = crop_width
               # all_bands, labels = random_crop(all_bands, labels, 20, 70)

        
        return all_bands, labels, fileName



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
    
def LoadData(batch_size, split, num_workers):
    return ReshapeDataLoader(DataLoader(GreenlandData(split=split),
                                        batch_size=batch_size,
                                        shuffle=(split=='train'),
                                        num_workers=num_workers,))
    
    
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
    Generate a bar plot of the class distrinution in the dataset and saves it.
    Args:
        dataset (DataLoader): The DataLoader instance that provides batches of size 1 of the data.
        path_to_plot (str): Path to the folder where the plot will be saved
        state (str, optional): State of the dataset used to name the plot. Defaults to 'train'.
    """
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
    model, _ = load_model(model, path_to_model, epoch)
    model = model.to(device)
    label_counter = Counter()
    pred_counter =  Counter()
    # Store true and predicted labels for confusion matrix
    all_true_labels = []
    all_pred_labels = []
    for idx, (data, labels, image_name) in enumerate(dataLoader):
        if idx == numImages:
            break
        unique_label, counts_label = np.unique(labels, return_counts=True)
        label_counter.update(dict(zip(unique_label, counts_label)))
        
        _, ax = plt.subplots(nrows=1, ncols=2, figsize = (20, 15))
  
        labels = labels.to(device)
        
        # plot ground truth
        ax[0].imshow(labels.squeeze(0).cpu().numpy(), cmap='tab20', vmin=0, vmax=len(label_names) - 1)
        ax[0].axis('off')
        ax[0].set_title('Ground Truth')

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
            ax[1].imshow(yhat.squeeze(0).cpu().numpy(), cmap='tab20', vmin=0, vmax=len(label_names) - 1)
            ax[1].axis('off')
            ax[1].set_title(image_name[0])
        plt.savefig(f'{path_to_plot}/{image_name[0]}.png')
    class_counts_labels = [label_counter.get(i, 0) for i in range(len(label_names))]
    class_counts_pred = [pred_counter.get(i, 0) for i in range(len(label_names))]
    fig, ax = plt.subplots(figsize=(20, 8))
    width = 0.35  # width of the bars
    x = np.arange(len(label_names))  # the label positions

    # Plot the bars for ground truth (blue) and predicted classes (green)
    ax.bar(x - width / 2, np.array(class_counts_labels)/sum(class_counts_labels), width, label='Ground Truth', color='blue', alpha=0.7)
    ax.bar(x + width / 2, np.array(class_counts_pred)/sum(class_counts_pred), width, label='Predicted', color='green', alpha=0.7)

    # Set the labels and titles
    ax.set_title('Class Frequency Comparison: Ground Truth vs. Predicted')
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45)
    ax.legend()
    plt.savefig(f'{path_to_plot}/Label_distribution_test.png')
    
    #Plot confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=np.arange(len(label_names)))

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca())
    plt.title('Confusion Matrix')
    plt.savefig(f'{path_to_plot}/Confusion_Matrix.png')
    


