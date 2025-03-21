from datetime import timedelta

from lightning.pytorch.callbacks import RichProgressBar

from cancer_classification.utils.mlflow_utils import ModelCheckpointWithCleanupCallback
from cancer_classification.custom_callbacks.timer_with_logging_callback import TimerWithLoggingCallback


def get_callbacks():
    checkpoint_callback = ModelCheckpointWithCleanupCallback(
        save_top_k=2,
        save_last=True,
        monitor="val_loss_epoch",
        mode="min",
        verbose=True,
        filename='{epoch}-{val_acc_epoch:.2f}'
    )

    timer_with_logging_callback = TimerWithLoggingCallback(duration=timedelta(weeks=1), interval='epoch')

    return [
        checkpoint_callback,
        timer_with_logging_callback,
        RichProgressBar()
    ]
