import torch
import rasterio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

print(torch.cuda.is_available())
seed = 323444           # the seed value used to initialise the random number generator of PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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



class Hypercolumn(nn.Module):

    def __init__(self):
        super(Hypercolumn, self).__init__()

        #TODO: define your architecture and forward pass here
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(483, 256, kernel_size=1, stride=1),           # 3 (input) + 32 + 64 + 128 + 256 = 487
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 7, kernel_size=1, stride=1)
        )


    def forward(self, x):
        #TODO
        upsample = nn.Upsample(size=(x.size(2), x.size(3)))
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        hypercol = torch.cat(
            (x, upsample(x1), upsample(x2), upsample(x3), upsample(x4)),
            dim=1)
        return self.final(hypercol)
    
from torch.utils.data import DataLoader
batch_size=3
dataset_train = GreenlandData(year=2014)
dataloader_train = DataLoader(GreenlandData(year=2014), batch_size=batch_size, num_workers=1)
model = Hypercolumn()

data, _ ,__= iter(dataloader_train).__next__()
print("Original data shape:", data.shape)

# Convert to float32 if needed
data = data.float()

# Ensure the shape is [batch_size, channels, height, width]
if data.ndim == 4 and data.shape[-1] != data.shape[1]:  # Likely [batch_size, height, width, channels]
    data = data.permute(0, 3, 1, 2)  # Rearrange to [batch_size, channels, height, width]

print("Adjusted data shape:", data.shape)
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
    for idx, (data, target, img_name) in enumerate(data_loader):

        data = data.float()

        # Ensure the shape is [batch_size, channels, height, width]
        if data.ndim == 4 and data.shape[-1] != data.shape[1]:  # Likely [batch_size, height, width, channels]
            data = data.permute(0, 3, 1, 2)  # Rearrange to [batch_size, channels, height, width]

        target = target.long()

        # Ensure the shape is [batch_size, channels, height, width]
        if target.ndim == 4 and target.shape[-1] != target.shape[1]:  # Likely [batch_size, height, width, channels]
            target = target.permute(0, 3, 1, 2)  # Rearrange to [batch_size, channels, height, width]
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

os.makedirs('cnn_states/Hypercolumn', exist_ok=True)

def load_model(epoch='latest'):
    model = Hypercolumn()
    modelStates = glob.glob('cnn_states/Hypercolumn/*.pth')
    if len(modelStates) and (epoch == 'latest' or epoch > 0):
        modelStates = [int(m.replace('cnn_states/Hypercolumn/','').replace('.pth', '')) for m in modelStates]
        if epoch == 'latest':
            epoch = max(modelStates)
        stateDict = torch.load(open(f'cnn_states/Hypercolumn/{epoch}.pth', 'rb'), map_location='cpu')
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
    for idx, (data, target, img_name) in enumerate(data_loader):
        with torch.no_grad():

        #TODO: likewise, implement the validation routine. This is very similar, but not identical, to the training steps.
            data = data.float()

        # Ensure the shape is [batch_size, channels, height, width]
            if data.ndim == 4 and data.shape[-1] != data.shape[1]:  # Likely [batch_size, height, width, channels]
                data = data.permute(0, 3, 1, 2)  # Rearrange to [batch_size, channels, height, width]\
            target = target.long()

            # Ensure the shape is [batch_size, channels, height, width]
            if target.ndim == 4 and target.shape[-1] != target.shape[1]:  # Likely [batch_size, height, width, channels]
                target = target.permute(0, 3, 1, 2)  # Rearrange to [batch_size, channels, height, width]
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
    torch.save(model.state_dict(), open(f'cnn_states/Hypercolumn/{epoch}.pth', 'wb'))
    
# define hyperparameters
device = 'cuda'
start_epoch = 0        # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
batch_size = 2
learning_rate = 0.1
weight_decay = 0.001
num_epochs = 10



# initialise data loaders
dataset_train_2014 = GreenlandData(year=2014)
dataset_train_2015 = GreenlandData(year=2015)
# Combine the datasets
combined_dataset = torch.utils.data.ConcatDataset([dataset_train_2014, dataset_train_2015])
# Create DataLoader
dl_train = DataLoader(combined_dataset, batch_size=batch_size,shuffle=True, num_workers=1)
dl_val = DataLoader(GreenlandData(year=2016), batch_size=batch_size,shuffle=True, num_workers=1)

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
    
with rasterio.open('data/images/train/2014/tile_159_2_2.tif') as src:
        # Read the RGB bands (3, 2, 1)
    rgb = np.dstack([src.read(3), src.read(2), src.read(1)])
        # Normalize RGB for better visualization
    rgb = rgb.astype(float)
    #rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    plt.figure()
    plt.imshow(rgb)

with rasterio.open('data/labels/train/tile_159_2_2.tif') as lbl_src:
    labels = lbl_src.read(1)
    plt.figure()
    plt.imshow(labels, cmap='tab20', vmin=0, vmax=len(label_names) - 1)