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

from cancer_classification.lightning_modules.custom_models.single_layer_ff import SingleLayerFF


class Base__CancerClassification__LightningModule(pl.LightningModule):
    def __init__(self, num_classes=9, learning_rate=1e-3):
        super(Base__CancerClassification__LightningModule, self).__init__()
        self.save_hyperparameters()

        # Define the ResNet model
        self.model = SingleLayerFF()
        self.num_classes = num_classes

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        self.val_loss_outputs = []
        self.train_loss_outputs = []

        # Define metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes)

        self.precision = MulticlassPrecision(num_classes=self.num_classes)
        self.recall = MulticlassRecall(num_classes=self.num_classes)
        self.f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.log('train_loss_batch', loss, prog_bar=True)
        self.train_loss_outputs.append(loss)

        self.train_accuracy.update(y_hat, y)

        return loss

    def on_train_epoch_end(self):
        train_loss_epoch = torch.stack(self.train_loss_outputs).mean()
        self.log('train_loss_epoch', train_loss_epoch)
        self.train_loss_outputs = []

        self.log('train_acc_epoch', self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.val_loss_outputs.append(loss)

        self.val_accuracy.update(y_hat, y)

        return loss
    
    def on_validation_epoch_end(self):
        val_loss_epoch = torch.stack(self.val_loss_outputs).mean()
        self.log('val_loss_epoch', val_loss_epoch)
        self.val_loss_outputs = []

        self.log('val_acc_epoch', self.val_accuracy.compute())
        self.val_accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.test_accuracy.update(y_hat.argmax(dim=1), y)
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
        self.log('test_accuracy', self.test_accuracy.compute())
        self.log('test_precision', self.precision.compute())
        self.log('test_recall', self.recall.compute())
        self.log('test_f1_score', self.f1_score.compute())

        _plot_and_log_confusion_matrix()


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.001)
        return optimizer
