import torch
import numpy as np
import segmentation_models_pytorch as smp

# Hyperparams
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
LOSS_WEIGHTS = [0.7, 0.3]

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_params = dict(
            pooling='avg',             
            dropout=0.2,               
            activation="sigmoid",      
            classes=2, # 2 types of lumps malign or benign                 
        )
model_backbone = "resnet34"
model = smp.UnetPlusPlus(model_backbone=model_backbone, encoder_weights="imagenet", decoder_attention_type=None, aux_params=model_params)
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

# Training Loop
for epoch in NUM_EPOCHS:
    pass