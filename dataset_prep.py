## Prep the dataset here
import os
import torch
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class CustomLungDataset(Dataset):
    def __init__(self, mhd_data_list, raw_data_list, annotation_csv):
        self.mhd_data_list = mhd_data_list
        self.raw_data_list = raw_data_list
        self.annotation_csv = annotation_csv

    def __len__(self):
        return len(self.mhd_data_list)
    
    # Find the annotation data from the annotation csv
    def find_annotation_data(self):
        pass
    
    # TODO Get the mask from the image return the mask and the image
    def process_image(self, mhd_data):
        pass
    
    def __getitem__(self, idx):
        pass

# Return the Dataloader from here
def prep_dataset(dataset_loc):
    dataset_loc = ""
    dataset_files = os.listdir(dataset_loc)
    image_info = []
    image_raw  = []

    # Sort into train
    for file in dataset_files:
        if file.endswith(".mhd"):
            image_info.append(file)
        elif file.endswith(".raw"):
            image_raw.append(file)

    # Read the data
    for info in image_info:
        # Read the iamge info
        with open(os.path.join(dataset_loc, info), 'r') as spec_image_info:
            data = spec_image_info.readlines()