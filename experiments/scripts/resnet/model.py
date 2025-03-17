import os

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
        self.log('val_acc', self.val_accuracy(y_hat, y))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy(y_hat, y))

        self.precision.update(y_hat.argmax(dim=1), y)
        self.recall.update(y_hat.argmax(dim=1), y)
        self.f1_score.update(y_hat.argmax(dim=1), y)
        self.confusion_matrix.update(y_hat.argmax(dim=1), y)

        return loss

    def on_test_epoch_end(self):
        def _plot_and_log_confusion_matrix():
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Compute confusion matrix
            cm = self.confusion_matrix.compute().cpu().numpy()

            # Plot confusion matrix
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            # Save confusion matrix plot as an image
            file_path = 'confusion_matrix.png'
            fig.savefig(file_path)

            # Log the file path
            self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path="confusion_matrix.png", artifact_path='plots')
            os.remove('confusion_matrix.png')

            plt.close()

        # Update precision, recall, and f1 metrics
        self.log('test_precision', self.precision.compute())
        self.log('test_recall', self.recall.compute())
        self.log('test_f1_score', self.f1_score.compute())

        _plot_and_log_confusion_matrix()


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
