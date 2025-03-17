from datetime import timedelta

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Timer


class TimerWithLoggingCallback(Timer):
    def __init__(self, duration=timedelta(weeks=1), interval='epoch'):
        super().__init__(duration=duration, interval=interval)


    def on_train_epoch_end(self, trainer, pl_module: LightningModule):
        pl_module.log('train_time', self.time_elapsed('train'), prog_bar=True)
