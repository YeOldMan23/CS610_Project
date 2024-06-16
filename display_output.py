import torch
import argparse
import random
import numpy as np
import cv2
import os

from .dataset_prep import prep_dataset

# Display the training of the model
def test_model(model, dataloader, threshold, save_results=False) -> None:
    with torch.no_grad():
        model.eval()
        # * Pull out 10 Random images from the test set to show the segmentation
        random_10_indices = random.sample([i for i in range(len(dataloader))], 10)
        random_10_images = []
        random_10_masks = []
        for index in random_10_indices:
            inputs, targets = dataloader[index]

            outputs = model(inputs.to(device)).detach().cpu().numpy()
            outputs[outputs >= threshold] = 1.0
            outputs[outputs < threshold] = 0.0

            # Reshape output to image
            for i in range(outputs.shape[0]):
                cur_mask = np.transpose(outputs[i, :, :, :], (1, 2, 0))
                cur_image = np.transpose(inputs[i, :, :, :], (1, 2, 0))

                random_10_images.append(cur_image)
                random_10_masks.append(cur_mask)

        # ? Show the image and mask side by side
        for i in range(10):
            cv2.imshow("Image", random_10_images[i])
            cv2.imshow("Mask", random_10_masks[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if save_results:
                cv2.imwrite(os.path.join(save_loc, "result_image_{}.png".format(i)), random_10_images[i])
                cv2.imwrite(os.path.join(save_loc, "result_mask_{}.png".format(i)), random_10_masks[i])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        help="Global Model path"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=0.5,
        help="Threshold for the model"
    )
    parser.add_argument(
        "--save_results",
        "-sr",
        action="store_true",
        default=False,
        help="Save the results of the model"
    )
    parser.add_argument(
        "--dataset_loc",
        "-dl",
        type=str,
        help="Location of dataset"
    )
    params = parser.parse_args()

    model_path = params.model_path
    save_loc = params.dataset_loc
    save_results = params.save_results
    threshold = params.threshold

    # ! Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path).to(device)

    # ! Load dataset
    train_dataloader, test_dataloader = prep_dataset(save_loc=save_loc)

    # ! test the model
    test_model(model, test_dataloader, threshold, save_results=save_results)



