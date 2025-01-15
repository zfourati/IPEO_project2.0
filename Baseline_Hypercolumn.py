"""
This script processes Greenland imagery to combine the rgb bands 
and apply them to a hypercolumn to predict landcover classification. 
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import glob
import Function_lib as lib

print('GPU available: ',torch.cuda.is_available())
seed = 323444           # the seed value used to initialise the random number generator of PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

path_to_model = 'cnn_states/HypercolumnBaseline'
os.makedirs(path_to_model, exist_ok=True)

path_to_plot = 'Plots/HypercolumnBaseline'
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

#Find the number of channels:
dataloader_train = DataLoader(lib.GreenlandData(split='train'), batch_size=1, num_workers=1)
data, _ , __= iter(dataloader_train).__next__()
nbr_channels = data.shape[3]

criterion = nn.CrossEntropyLoss()

#Load training and validation data
dl_train = lib.LoadData(batch_size, split='train', num_workers=1)
dl_val = lib.LoadData(batch_size, split='val', num_workers=1)

# load model
model, epoch = lib.load_model(lib.Hypercolumn(input_channels=nbr_channels), path_to_model, epoch=start_epoch)
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
dl_test = lib.LoadData(batch_size, split='test', num_workers=1)
loss_test, oa_test = lib.validate_epoch(dl_test, model, device)
print('Testing:  Loss: {:.2f}  OA: {:.2f}'.format(loss_test, 100*oa_test))

#Visualize predictions and label class distribution
dl_test_single = lib.LoadData(batch_size=1, split='test', num_workers=1)                
lib.visualize(dl_test_single,model, path_to_plot, path_to_model)

dl_train_single = lib.LoadData(batch_size, split='train', num_workers=1)
dl_val_single = lib.LoadData(batch_size, split='val', num_workers=1)
lib.plot_label_distribution(dl_train_single, path_to_plot)
lib.plot_label_distribution(dl_train_single,path_to_plot,  state = 'val')
