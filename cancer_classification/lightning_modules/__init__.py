from enum import Enum

from cancer_classification.lightning_modules.base_cancer_classification_lightning_module import Base__CancerClassification__LightningModule
from cancer_classification.lightning_modules.resnet18_cancer_classification_lightning_module import Resnet18__CancerClassification__LightningModule


model_classes = {
    "base": Base__CancerClassification__LightningModule,
    "resnet18": Resnet18__CancerClassification__LightningModule,
}
