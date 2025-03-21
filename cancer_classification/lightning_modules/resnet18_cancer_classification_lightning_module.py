import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)
import lightning.pytorch as pl

from torchvision.models import resnet18

from cancer_classification.lightning_modules.base_cancer_classification_lightning_module import (
    Base__CancerClassification__LightningModule
)


class Resnet18__CancerClassification__LightningModule(Base__CancerClassification__LightningModule):
    def __init__(self, model=resnet18(pretrained=False)):
        super(Base__CancerClassification__LightningModule, self).__init__()
        self.save_hyperparameters()

        # Define the ResNet model
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
