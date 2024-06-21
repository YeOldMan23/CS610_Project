## Prep the dataset here
import os
import cv2
import SimpleITK
import shutil
import numpy as np
from collections import namedtuple

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
def convert_data_to_image_mask_list(dataset_loc, candidates_loc, annotations_loc, save_loc) -> list:
    """
    @brief Converts the dataset and annotations into images and masks, save the dataset stats
    @param dataset_loc global location of dataset
    @param candidates_loc global location of candidates
    @param annotations_loc global location of annotations
    @param save_loc save location of candidates, global path

    return (list image_data, list image_mask) Image Data (512x512 images) and corresponding Mask
            lists are in order 
    """
    # Read the CSV file
    # Dateset is made up of subset of datasets
    dataset_files = os.listdir(dataset_loc)
    candidates_data = pd.read_csv(candidates_loc)
    annotations_data = pd.read_csv(annotations_loc)
    save_location = save_loc

    if os.path.exists(save_location):
        shutil.rmtree(save_location)
    os.mkdir(save_location)

    # Open the missing data
    with open(os.path.join(dataset_loc, "missing.txt")) as f:
        missing_uids = {uid.split('\n')[0] for uid in f}

    print("Missing Data : {}".format(missing_uids))

    # Make the image and the annotations for the dataset
    if os.path.exists(os.path.join(save_location)):
        shutil.rmtree(os.path.join(save_location))
    os.mkdir(os.path.join(save_location))
    os.mkdir(os.path.join(save_location, "images"))
    os.mkdir(os.path.join(save_location, "masks"))

    image_counter = 0
    annotation_counter = 0

    for dataset in dataset_files: # Subset 1-10
        if dataset.endswith(".csv") or dataset.endswith(".txt"):
            continue
        for data_file in os.listdir(os.path.join(dataset_loc, dataset)):
            # Get all the .mhd data
            if data_file.endswith(".mhd"):
                # Read the mhd data 
                data_file = os.path.join(dataset_loc, dataset, data_file)
                mhd_file = SimpleITK.ReadImage(data_file)
                ct_scan = np.array(SimpleITK.GetArrayFromImage(mhd_file), dtype=np.float32)
                ct_scan.clip(-1000, 1000, ct_scan)

                # Normalize the CT Scan from 0 to 1, float to display as a map
                ct_scan = (ct_scan - (-1000)) / 2000.0
                
                print("CT Scan Shape : {}".format(ct_scan.shape))

                origin_xyz = mhd_file.GetOrigin()
                voxel_size_xyz = mhd_file.GetSpacing()

                origin_xyz_np = np.array(origin_xyz)
                voxel_size_xyz_np = np.array(voxel_size_xyz)
                direction_matrix = np.array(mhd_file.GetDirection()).reshape(3, 3)

                # Get the annotation and candidate data from the image
                annotation_rows = annotations_data[annotations_data["seriesuid"] == data_file.rstrip(".mhd")]
                candidates_rows = candidates_data[candidates_data["seriesuid"] == data_file.rstrip(".mhd")]

                # Mask array
                ct_scan_mask = np.zeros(ct_scan.shape)

                # Find and choose the candidates that are malign as detections, all other nodules are not considered
                # Choose from annotations
                # ? Annotations contain all the annotations
                # ? Candidates contain all the candidates that are chosen
                diameters = {}
                candidates = []

                for _, row in annotation_rows.iterrows():
                    center_xyz = (row.coordX, row.coordY, row.coordZ)
                    diameters.setdefault(row.seriesuid, []).append(
                        (center_xyz, row.diameter_mm)
                    )

                CandidateInfoTuple = namedtuple(
                    'CandidateInfoTuple',
                    ['is_nodule', 'diameter_mm', 'series_uid', 'center_xyz']
                )
                
                # For each candidate, check if it is a nodule or not
                for _, row in candidates_rows.iterrows():
                    candidate_center_xyz = (row.coordX, row.coordY, row.coordZ)

                    candidate_diameter = 0.0

                    for annotation in diameters.get(row.seriesuid, []):
                        annotation_center_xyz, annotation_diameter = annotation

                        for i in range(3):
                            delta = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                            if delta > annotation_diameter / 4:
                                break
                            else:
                                candidate_diameter = annotation_diameter
                                break

                        candidates.append(CandidateInfoTuple(
                            bool(row['class']),
                            candidate_diameter,
                            row.seriesuid,
                            candidate_center_xyz
                        ))

                # Make sure candidates are clean
                candidates_clean = list(filter(lambda x: x.series_uid not in missing_uids, candidates))
                print("Candidates Clean : {}".format(candidates_clean))
                
                # Iterate over the candidates and create the mask
                for candidate in candidates_clean:
                    if candidate.is_nodule:
                        # Create the mask
                        center_xyz = candidate.center_xyz
                        
                        # Adjust the coordinates
                        origin_xyz = mhd_file.GetOrigin()
                        voxel_size_xyz = mhd_file.GetSpacing()
                        direction_matrix = np.array(mhd_file.GetDirection()).reshape(3, 3)

                        origin_xyz_np = np.array(origin_xyz)
                        voxel_size_xyz_np = np.array(voxel_size_xyz)

                        cri = ((center_xyz - origin_xyz_np) @ np.linalg.inv(direction_matrix)) / voxel_size_xyz_np
                        cri = np.round(cri)

                        print("Center : {}, Origin : {}, Voxel : {}, Direction : {}".format(center_xyz, origin_xyz, voxel_size_xyz, direction_matrix))

                        candidate_diameter = candidate.diameter_mm
                        print("Possible Nodule : {}  with diameter {}".format(center_xyz, candidate_diameter))

                        # Create the mask from the the candidates
                        mask = np.zeros(ct_scan.shape)
                        mask[int(center_xyz[0] - candidate_diameter):int(center_xyz[0] + candidate_diameter), \
                             int(center_xyz[1] - candidate_diameter):int(center_xyz[1] + candidate_diameter), \
                            int(center_xyz[2] - candidate_diameter):int(center_xyz[2] + candidate_diameter)] = 1

                        # Add the mask to the mask array
                        ct_scan_mask += mask

                # Slice out a section of the image and the mask for the dataset, slice grayscale
                for i in range(0, ct_scan.shape[0], 5):
                    image = ct_scan[i, :, :]
                    mask = ct_scan_mask[i, :, :]

                    # If the mask is empty, skip the image
                    if np.sum(mask) == 0:
                        continue

                    # Save the image and the mask
                    image_loc = os.path.join(save_location, "images", "image_{}.png".format(image_counter))
                    mask_loc = os.path.join(save_location, "masks", "mask_{}.png".format(image_counter))

                    # Save the image and the mask
                    print("Saving Image : {}, Mask : {}".format(image_loc, mask_loc))
                    cv2.imwrite(image_loc, image)
                    cv2.imwrite(mask_loc, mask)

                    image_counter += 1

                


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
        help="Global base location of dataset"
    )
    parser.add_argument(
        "--save_loc",
        "-sl",
        type=str,
        help="Global Save Location of dataset"
    )
    params = parser.parse_args()
    
    base_location = params.base_loc
    candidates_loc = os.path.join(base_location, "candidates.csv")
    annotations_loc = os.path.join(base_location, "annotations.csv")
    save_loc = params.save_loc
    data_prepped = False # ! Make false if dataset is prepped
    if not data_prepped:
        convert_data_to_image_mask_list(base_location, candidates_loc, annotations_loc, save_loc)