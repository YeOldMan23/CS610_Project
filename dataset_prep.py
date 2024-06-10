## Prep the dataset here
import os
import cv2
import SimpleITK
import shutil
import numpy as np

# Annotations
import pandas as pd
from PIL import Image

# Dataset Prep
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

# TODO : Convert All dataset into image data and masks using simply ITK
def convert_data_to_image_mask_list(dataset_loc, candidates_loc, save_loc):
    """
    @brief Converts the dataset and annotations into images and masks
    @param dataset_loc global location of dataset
    @param candidates_loc global location of candidates
    @param save_loc save location of candidates, relative path

    return (list image_data, list image_mask) Image Data (512x512 images) and corresponding Mask
            lists are in order 
    """
    # Read the CSV file
    # Dateset is made up of subset of datasets
    dataset_files = os.listdir(dataset_loc)
    candidates_data = pd.read_csv(candidates_loc)
    save_location = os.path.join(os.getcwd(), save_loc)

    if os.path.exists(save_location):
        shutil.rmtree(save_location)
    os.mkdir(save_location)
    
    image_data = []

    for dataset in dataset_files: # Subset 1-10
        for data_file in os.listdir(os.path.join(dataset_loc, dataset)):
            # Get all the .mhd data
            if data_file.endswith(".mhd"):
                # Read the mhd data 
                mhd_file = SimpleITK.ReadImage(data_file)
                ct_scan = np.array(SimpleITK.GetArrayFromImage(mhd_file), dtype=np.float32)
                ct_scan.clip(-1000, 1000, ct_scan)
                origin_xyz = mhd_file.GetOrigin()



class CustomLungDataset(Dataset):
    def __init__(self, mask_list, image_list):
        """
        @brief Custom Dataset for dataset
        @param mask_list list of mask information
        @param image_list list of image information
        """
        self.mask_list      = mask_list
        self.raw_data_list  = image_list

        # TODO : Transforms are the image augmentation
        self.transforms     = transforms.Compose([
            transforms.ToTensor(), # Convert to Tensor and Normalize [0, 1] (/255)
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20)
        ])

    def __len__(self):
        return len(self.mask_list)
    
    def __getitem__(self, idx):
        image, mask = self.image_list[idx], self.mask_list[idx]

        # Convet the image to PIL Image first
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

# Return the Dataset from here
def prep_dataset(dataset_loc, is_test=False):
    pass

if __name__ == '__main__':
    base_location = "C:\Users\kiere\Desktop\SMU MITB\CS610\LUNA16" # ! Replace with your own location
    candidates_loc = "C:\Users\kiere\Desktop\SMU MITB\CS610\LUNA16\candidates.csv" # ! Replace with your own location
    save_loc = ""
    data_prepped = True # ! Make false if dataset is prepped
    if not data_prepped:
        convert_data_to_image_mask_list(base_location, candidates_loc, save_Loc)