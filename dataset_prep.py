## Prep the dataset here
import os
import cv2
from sklearn.model_selection import train_test_split

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


