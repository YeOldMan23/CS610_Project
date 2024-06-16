import torch
import numpy as np
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
import math
import argparse

# Hyperparams
LEARNING_RATE = 0.001
LOSS_WEIGHTS = [0.7, 0.3]

from .dataset_prep import prep_dataset

# Loss Function
class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        alpha=0.6,
        beta=0.4,
        smooth=1,
        gamma=4 / 3,
        class_weights=None,
    ):
        super(FocalTverskyLoss, self).__init__()
        self.name = "Focal Tversky Loss"
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
        self.class_weights = class_weights

        # Keep BG weight at 1
        if self.class_weights is not None:
            self.class_weights[-1] = 1.0

            self.no_class = len(class_weights)

    def forward(self, inputs, targets):
        # Ok for multiclass multilabel since the entire image becomes the same dimensions
        class_input = inputs.view(-1)
        class_target = targets.view(-1)

        # True Positive, False Negative
        TP = (class_input * class_target).sum()
        FP = ((1 - class_target) * class_input).sum()
        FN = (class_target * (1 - class_input)).sum()

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        FocalTversky = (1 - Tversky) ** (1 / self.gamma)

        return FocalTversky
    
loss_functions = [FocalTverskyLoss(), torch.nn.BCELoss()]

def apply_loss(predictions, target):
    final_loss = 0
    for loss_func, weight in zip(loss_functions, LOSS_WEIGHTS):
        final_loss += loss_func(predictions, target)
    return final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc",
                        '-dl',
                        type=str,
                        help="Global Location of dataset")
    parser.add_argument("--save_loc",
                        "-sl",
                        type=str,
                        help="Save location for the model")
    parser.add_argument("--candidates_loc",
                        '-cl',
                        type=str,
                        help="Global Location for the candidates.csv location")
    parser.add_argument("--batch_size",
                        '-b',
                        type=int,
                        default=4,
                        help="Batch Size")
    parser.add_argument("--epochs",
                        'e',
                        type=int,
                        default=50,
                        help="Number of epochs to train")
    parser.add_argument("--save_type",
                        '-save',
                        type=str,
                        default='loss',
                        help="Metric to determine which metric to save model by, \
                              choose from [loss, IoU, recall, precision]")
    params = parser.parse_args()

    base_location = params.data_loc
    candidates_loc = params.candidates_loc 
    save_loc = params.save_loc
    NUM_EPOCHS = params.epochs
    BATCH_SIZE = params.batch_size
    save_type = params.save_type

    assert save_type in ["loss", "IoU", "recall", "precision"], "Please choose correct save type"

    # Import Dataset Here, make dataloader
    train_dataloader, test_dataloader = prep_dataset(save_loc=save_loc)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = dict(
                pooling='avg',             
                dropout=0.2,               
                activation="sigmoid",      
                classes=2, # 2 types of lumps malign (1) or benign(0)            
            )
    model_backbone = "resnet34"
    model = smp.UnetPlusPlus(model_backbone=model_backbone, 
                            encoder_weights="imagenet", 
                            decoder_attention_type=None, 
                            aux_params=model_params, 
                            activation="softmax")
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(),LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=0, threshold=0.01, verbose=True)
    
    # ! Metrics to consider before saving the model
    best_loss = -math.inf
    best_IoU = 0

    for epoch in NUM_EPOCHS:
        print("Epoch : {}".format(epoch + 1))
        # Model training and metrics
        model.train()

        # Metrics in terms of pixel
        train_epoch_loss = 0
        train_epoch_IoU = 0
        train_epoch_precision = 0
        train_epoch_recall = 0 

        for idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = apply_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()

            # Calculate other stuff
            outputs = outputs.view(-1)
            targets = targets.view(-1)
            true_positive = (outputs * targets)
            false_positive = (outputs) * (1 - targets)
            false_negative = (1 - outputs) * targets
            true_negative = (1 - outputs) * (1 - targets)

            IoU = true_positive / (true_positive + false_positive + false_negative)
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            train_epoch_IoU += IoU
            train_epoch_precision += precision
            train_epoch_recall += recall

        # * Print params here
        ave_train_loss = train_epoch_loss / len(train_dataloader)
        ave_train_IoU = train_epoch_IoU / len(train_dataloader)
        ave_train_precision = train_epoch_precision / len(train_dataloader)
        ave_train_recall = train_epoch_recall / len(train_dataloader)

        print(f"Loss : {ave_train_loss} IoU : {ave_train_IoU} Precision : {ave_train_precision} Recall : {ave_train_recall}")
        scheduler.step(ave_train_loss)

        test_epoch_loss = 0
        test_epoch_IoU = 0
        test_epoch_precision = 0
        test_epoch_recall = 0

        model.eval()
        # Put against the test set
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(tqdm(test_dataloader)):
                outputs = model(inputs.to(device))
                loss = apply_loss(outputs, targets)
                
                test_epoch_loss += loss.item()

                # Calculate other stuff
                true_positive = outputs * targets
                false_positive = (outputs) * (1 - targets)
                false_negative = (1 - outputs) * targets
                true_negative = (1 - outputs) * (1 - targets)

                IoU = true_positive / (true_positive + false_positive + false_negative)
                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative) 

                test_epoch_IoU += IoU
                test_epoch_precision += precision
                test_epoch_recall += recall

            # * Print params here
            ave_test_loss = train_epoch_loss / len(train_dataloader)
            ave_test_IoU = train_epoch_IoU / len(train_dataloader)
            ave_test_precision = train_epoch_precision / len(train_dataloader)
            ave_test_recall = train_epoch_recall / len(train_dataloader)

            print(f"Loss : {ave_test_loss} IoU : {ave_test_IoU} Precision : {ave_test_precision} Recall : {ave_test_recall}")

            # Save accordingly
            if ave_test_IoU > best_IoU:
                print("Saving IoU model...")
                torch.save(model, os.path.join(save_loc, "BestIOUModel.pt"))
            
            if ave_test_loss < best_loss:
                print("Saving best loss")
                torch.save(model, os.path.join(save_loc, "BestLossModel.pt"))