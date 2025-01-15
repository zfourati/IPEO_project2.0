#  Landcovermappingin Greenland with multi-temporal imagery
## Introduction


## Table of content

## Installation
To set up the Greenland_ipeo_venv environment, use the provided environment.yaml file.

## Project structure
- `Baseline_Hypercolumn.py` : Script to process Greenland imagery to combine the RGB bands and apply them to a hypercolumn in order to predict landcover classification.
- `Baseline_Hypercolumn_DA.py` : Script to process Greenland imagery to combine the RGB bands, apply data augmentation and apply them to a hypercolumn in order to predict landcover classification.
- `Hypercolumn_features.py` : Script to process Greenland imagery to combine temporal statistics for landcover classification. It computes features such as NDVI, NDWI, and NDSI, along with their mean and standard deviation over multiple years  (2014, 2015, 2016 for training/validation and 2023 for testing). These temporal features are integrated into 
a hypercolumn representation to predict landcover classification.
- `Baseline_Unet.py`:

## Usage
For each experiment run the corresponding python file in order to:
- Load the dataset
- Load the model and the existing state dict
- Train the model
- Validate the model
- Test the model
- Generate and save a bar plot of the class distribution in the training, validation and test dataset
- Generate and save a plot of the Groundtruth vs. prediction for a number of images in the test dataset
- Generate and save a bar plot of the Label distribution of groundtruth and the predictions of the dataset.
- Plot and save the confusion matrix
## Data
The dataset used for training can be downloaded from [[link-to-dataset](https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg)]. Unzip the folder and place the labels/ and images/ folders in the data/ folder.
## Model

## Evaluation
