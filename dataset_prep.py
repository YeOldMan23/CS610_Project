## Prep the dataset here
import os
import cv2
import SimpleITK
import shutil
import numpy as np

import argparse

# Annotations
import pandas as pd
from PIL import Image

# Dataset Prep
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

# TODO : Convert All dataset into image data and masks using simply ITK
def convert_data_to_image_mask_list(dataset_loc, candidates_loc, annotations_loc, save_loc):
    """
    @brief Converts the dataset and annotations into images and masks
    @param dataset_loc global location of dataset
    @param candidates_loc global location of candidates
    @param annotations_loc global location of annotations
    @param save_loc save location of candidates, relative path

    return (list image_data, list image_mask) Image Data (512x512 images) and corresponding Mask
            lists are in order 
    """
    # Read the CSV file
    # Dateset is made up of subset of datasets
    dataset_files = os.listdir(dataset_loc)
    candidates_data = pd.read_csv(candidates_loc)
    annotations_data = pd.read_csv(annotations_loc)
    save_location = os.path.join(os.getcwd(), save_loc)

    if os.path.exists(save_location):
        shutil.rmtree(save_location)
    os.mkdir(save_location)
    
    image_data = []

    for dataset in dataset_files: # Subset 1-10
        if dataset.endswith(".csv"):
            continue
        for data_file in os.listdir(os.path.join(dataset_loc, dataset)):
            # Get all the .mhd data
            if data_file.endswith(".mhd"):
                # Read the mhd data 
                mhd_file = SimpleITK.ReadImage(data_file)
                ct_scan = np.array(SimpleITK.GetArrayFromImage(mhd_file), dtype=np.float32)
                ct_scan.clip(-1000, 1000, ct_scan)

                # Normalize the CT Scan from 0 to 1, float to display as a map
                ct_scan = (ct_scan - (-1000)) / 2000.0

                # Get the annotation and candidate data from the image
                annotation_rows = annotations_data[annotations_data["seriesuid"] == data_file.rstrip(".mhd")]
                candidates_rows = candidates_data[candidates_data["seriesuid"] == data_file.rstrip(".mhd")]

                # Mask array
                ct_scan_mask = np.zeros(ct_scan.shape)

                # Find and choose the candidates that are malign as detections, all other nodules are not considered
                # Choose from annotations
                # ? Annotations contain all the annotations
                # ? Candidates contain all the candidates that are chosen


class CustomLungDataset(Dataset):
    def __init__(self, mask_list, image_list, dataset_stats : tuple, is_train=False):
        """
        @brief Custom Dataset for dataset
        @param mask_list list of mask information
        @param image_list list of image information
        """
        self.mask_list      = mask_list
        self.raw_data_list  = image_list
        self.is_train = is_train
        self.dataset_mean, self.dataset_std = dataset_stats

        # TODO : Transforms are the image augmentation
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(), # Convert to Tensor and Normalize [0, 1] (/255)
            transforms.Normalize([self.dataset_mean, ], [self.dataset_std, ]),
            transforms.RandomHorizontalFlip(0.25),
            transforms.RandomVerticalFlip(0.25),
            transforms.RandomRotation(20)

        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([self.dataset_mean, ], [self.dataset_std, ])
        ])

    def __len__(self):
        return len(self.mask_list)
    
    def __getitem__(self, idx):
        image, mask = self.image_list[idx], self.mask_list[idx]

        # Convet the image to PIL Image first
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.is_train:
            image, mask = self.train_transforms(image, mask)
        else:
            image, mask = self.test_transforms(image, mask)

        return image, mask

# Return the Dataloader(s) from here
def prep_dataset(save_loc) -> list:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_loc",
        "-bl",
        type=str,
        default= "C:\Users\kiere\Desktop\SMU MITB\CS610\LUNA16",
        help="Global base location of dataset"
    )
    parser.add_argument(
        "--save_loc",
        "-sl",
        type=str,
        default="dataset_save",
        help="Relative Save Location of dataset"
    )
    params = parser.parse_args()
    
    base_location = params.base_loc
    candidates_loc = os.path.join(base_location, "candidates.csv")
    annotations_loc = os.path.join(base_location, "annotations.csv")
    save_loc = os.path.join(os.getcwd(), params.save_loc)
    data_prepped = True # ! Make false if dataset is prepped
    if not data_prepped:
        convert_data_to_image_mask_list(base_location, candidates_loc, annotations_loc, save_loc)