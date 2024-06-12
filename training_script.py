import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Hyperparams
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
LOSS_WEIGHTS = [0.7, 0.3]

from .dataset_prep import prep_dataset

base_location = "C:\Users\kiere\Desktop\SMU MITB\CS610\LUNA16" # ! Replace with your own location
candidates_loc = "C:\Users\kiere\Desktop\SMU MITB\CS610\LUNA16\candidates.csv" # ! Replace with your own location
save_loc = "dataset_save"

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=0, threshold=0.01, verbose=True)

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

# Import Dataset Here, make dataloader
train_dataloader, test_dataloader = prep_dataset(save_loc=save_loc)

# Training Loop
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
        true_positive = outputs * targets
        false_positive = (outputs) * (1 - targets)
        false_negative = (1 - outputs) * targets
        true_negative = (1 - outputs) * (1 - targets)

        IoU = true_positive / (true_positive + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        
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

