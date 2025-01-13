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
        # Open the Landsat image and read the required bands
        with rasterio.open(imgName) as src:
            # Read the first six bands
            bands = np.dstack([src.read(i) for i in range(1, 7)])
            
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
        
        # Data augmentation (applied to both images and labels)
        if self.augment:
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
            
            if random.random() > 0.5:
                crop_width = np.randint(50, 110)
                crop_height = crop_width
                all_bands, labels = random_crop(all_bands, labels, 20, 70)

        # Apply transforms if any
        if self.transforms is not None:
            all_bands = self.transforms(all_bands)
        
        # Open and read the label data
        with rasterio.open(labelName) as lbl_src:
            labels = lbl_src.read(1)  # Read the first band which contains the labels
        
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


model = Unet(
    encoder_name="resnet34",        # Use a ResNet-34 encoder
    encoder_weights="imagenet",    # Pre-trained on ImageNet
    in_channels=9,                 # Number of input channels (e.g., RGB images)
    classes=7                      # Number of output classes
)   

batch_size=3
dataset_train = GreenlandData(year=2014)
dataloader_train = ReshapeDataLoader(DataLoader(GreenlandData(year=2014), batch_size=batch_size, num_workers=2))
model = Hypercolumn()
data, _ , __= iter(dataloader_train).__next__()

pred = model(data)


assert pred.size(1) == len(dataset_train.LABEL_CLASSES), f'ERROR: invalid number of model output channels (should be # classes {len(dataset_train.LABEL_CLASSES)}, got {pred.size(1)})'
assert pred.size(2) == data.size(2), f'ERROR: invalid spatial height of model output (should be {data.size(2)}, got {pred.size(2)})'
assert pred.size(3) == data.size(3), f'ERROR: invalid spatial width of model output (should be {data.size(3)}, got {pred.size(3)})'


criterion = nn.CrossEntropyLoss()
from torch.optim import SGD

def setup_optimiser(model, learning_rate, weight_decay):
    return SGD(
        model.parameters(),
        learning_rate,
        weight_decay
    )
    
from tqdm.notebook import trange      # pretty progress bar


def train_epoch(data_loader, model, optimiser, device):

    # set model to training mode. This is important because some layers behave differently during training and testing
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

        #print loss
        #print('loss:', loss.item())
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

    # normalise stats
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return model, loss_total, oa_total

import glob


def load_model(epoch='latest'):
    model = Hypercolumn()
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

def validate_epoch(data_loader, model, device):       # note: no optimiser needed

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

    # normalise stats
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return loss_total, oa_total


def save_model(model, epoch):
    torch.save(model.state_dict(), open(f'{path_to_model}/{epoch}.pth', 'wb'))
    
# define hyperparameters
device = 'cuda'
start_epoch = 'latest'        # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
batch_size = 30
learning_rate = 0.1
weight_decay = 0.001
num_epochs = 10

validation_split_ratio = 0.2


# initialise data loaders
dataset_train_2014 = GreenlandData(year=2014)
dataset_train_2015 = GreenlandData(year=2015)
dataset_train_2016 = GreenlandData(year=2016)
# Combine the datasets
combined_dataset = torch.utils.data.ConcatDataset([dataset_train_2014, dataset_train_2015,dataset_train_2016])
validation_size = int(validation_split_ratio * len(combined_dataset))
training_size = len(combined_dataset) - validation_size
# Shuffle and split the dataset
train_dataset, val_dataset = random_split(combined_dataset, [training_size, validation_size])

# Create DataLoaders for training and validation
dl_train = ReshapeDataLoader(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
dl_val = ReshapeDataLoader(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))


# load model
model, epoch = load_model(epoch=start_epoch)
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
    save_model(model, epoch)
    
#Testing
dataset_test = GreenlandData(year=2023)
dl_test = ReshapeDataLoader(DataLoader(dataset_test, batch_size=batch_size, shuffle=False))
loss_test, oa_test = validate_epoch(dl_test, model, device)
print('Testing:  Loss: {:.2f}  OA: {:.2f}'.format(loss_test, 100*oa_test))



#Plot results
def plot_label_distribution(dataset, state = 'train'):
    # Initialize a counter for label frequencies
    label_counter = Counter()

    # Iterate over the dataset
    for _, labels, _ in dataset:
        unique, counts = np.unique(labels, return_counts=True)
        label_counter.update(dict(zip(unique, counts)))

    # Map label indices to class names
    class_names = label_names
    class_counts = [label_counter.get(i, 0) for i in range(len(class_names))]
    #print(sum(class_counts))
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

def visualize(dataLoader, epoch = 'latest', numImages=5):
    model, _ = load_model(epoch)
    model = model.to(device)
    label_counter = Counter()
    pred_counter =  Counter()
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
        
#Visualize predictions and label class distribution
dl_test_single = ReshapeDataLoader(DataLoader(dataset_test, batch_size=1, shuffle=False))                  
visualize(dl_test_single)

dl_train_single = ReshapeDataLoader(DataLoader(train_dataset, batch_size=1, shuffle=False))
dl_val_single = ReshapeDataLoader(DataLoader(val_dataset, batch_size=1, shuffle=False))
plot_label_distribution(dl_train_single)
plot_label_distribution(dl_train_single, state = 'val')