import io

from PIL import Image

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


class ResNet18__LightningModule(pl.LightningModule):
    def __init__(self, num_classes=9, learning_rate=1e-3):
        super(ResNet18__LightningModule, self).__init__()
        self.save_hyperparameters()

        # Define the ResNet model
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)

        self.precision = MulticlassPrecision(num_classes=num_classes)
        self.recall = MulticlassRecall(num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

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
        def _plot_and_log_confusion_matrix():
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Compute confusion matrix
            cm = self.confusion_matrix.compute().cpu().numpy()

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            # Log confusion matrix plot to the logger
            self.logger.experiment.log_figure(plt, "Confusion Matrix.png")

            plt.close()

        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy(y_hat, y))

        # Update precision, recall, and f1 metrics
        self.log('test_precision', self.precision(y_hat, y))
        self.log('test_recall', self.recall(y_hat, y))
        self.log('test_f1_score', self.f1_score(y_hat, y))

        _plot_and_log_confusion_matrix()

        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
