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
from tqdm.notebook import trange
import random
from torch.optim import SGD
import glob
from segmentation_models_pytorch import Unet
import segmentation_models_pytorch as smp
from Function_lib_v2_features import *

print(torch.cuda.is_available())
seed = 323444           # the seed value used to initialise the random number generator of PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

path_to_model = 'cnn_states/pretrainedUnet_features'
os.makedirs(path_to_model, exist_ok=True)

path_to_plot = 'Plots/pretrainedUnet_features'
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

model = Unet(
   encoder_name="resnet34",        # Use a ResNet-34 encoder
   encoder_weights="imagenet",    # Pre-trained on ImageNet
   in_channels=3,                 # Number of input channels (e.g., RGB images)
    classes=7                      # Number of output classes
)

"""
batch_size=30
dataset_train = GreenlandData(split='train')
dataloader_train = LoadData(batch_size, 'train', num_workers=1) #ReshapeDataLoader(GreenlandData(split='train'), batch_size=batch_size, num_workers=2)
data, target, img_name = iter(dataloader_train).__next__()

print(np.shape(data))

pred = model(data)

print(f"Model Output Shape: {pred.shape}")

loss = criterion(pred, target)
print(loss)

# backward pass
loss.backward()

assert pred.size(1) == len(dataset_train.LABEL_CLASSES), f'ERROR: invalid number of model output channels (should be # classes {len(dataset_train.LABEL_CLASSES)}, got {pred.size(1)})'
assert pred.size(2) == data.size(2), f'ERROR: invalid spatial height of model output (should be {data.size(2)}, got {pred.size(2)})'
assert pred.size(3) == data.size(3), f'ERROR: invalid spatial width of model output (should be {data.size(3)}, got {pred.size(3)})'

"""

criterion = nn.CrossEntropyLoss()

#criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
 # for image segmentation dice loss could be the best first choice
        #self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)


dl_train = LoadData(batch_size, split='train', num_workers=1)
dl_val = LoadData(batch_size, split='val', num_workers=1)

# load model
model, epoch = load_model(model, path_to_model, epoch=start_epoch)
optim = setup_optimiser(model, learning_rate, weight_decay)


# do epochs
while epoch < num_epochs:
    print(epoch)

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
dl_test = LoadData(batch_size, split='test', num_workers=1)
loss_test, oa_test = validate_epoch(dl_test, model, device)
print('Testing:  Loss: {:.2f}  OA: {:.2f}'.format(loss_test, 100*oa_test))

#Visualize predictions and label class distribution
dl_test_single = LoadData(batch_size=1, split='test', num_workers=1)                
visualize(dl_test_single, model, path_to_plot, path_to_model)

dl_train_single = LoadData(batch_size, split='train', num_workers=1)
dl_val_single = LoadData(batch_size, split='val', num_workers=1)
plot_label_distribution(dl_train_single, path_to_plot)
plot_label_distribution(dl_train_single,path_to_plot,  state = 'val')

