import torch.nn as nn

from torchvision.models import resnet18

from cancer_classification.lightning_modules.base_cancer_classification_lightning_module import (
    Base__CancerClassification__LightningModule
)


class Resnet18__CancerClassification__LightningModule(Base__CancerClassification__LightningModule):
    def __init__(self):
        super().__init__()

        # Define the ResNet model
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
