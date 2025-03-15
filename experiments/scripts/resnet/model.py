import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning.pytorch as pl
from torchvision.models import resnet18


class ResNet18__LightningModule(pl.LightningModule):
    def __init__(self, num_classes=9, learning_rate=1e-3):
        super(ResNet18__LightningModule, self).__init__()
        self.save_hyperparameters()

        # Define the ResNet model
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(y_hat, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy(y_hat, y))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
