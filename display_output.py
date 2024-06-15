import torch
import random
import numpy as np

from .dataset_prep import prep_dataset


# ! Load the model
model_path = ""
model = torch.load(model_path)

# ! Load dataset
save_loc = ""
train_dataloader, test_dataloader = prep_dataset(save_loc=save_loc)


# Display the training of the model
with torch.no_grad():
    model.eval()
    # * Pull out 10 Random images from the test set to show the segmentation
    random_10_indices = random.sample([i for i in range(len(test_dataloader))], 10)
    random_10_images = []
    random_10_masks = []
    for index in random_10_indices:
        inputs, targets = test_dataloader[index]

        outputs = model(inputs).detach().cpu().numpy()

        # Reshape output to image
        for i in range(outputs.shape[0]):
            cur_mask = np.transpose(outputs[i, :, :, :], (1, 2, 0))
            cur_image = np.transpose(inputs[i, :, :, :], (1, 2, 0))

        # Show image and mask side by side 