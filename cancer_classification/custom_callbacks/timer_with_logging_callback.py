from datetime import timedelta

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Timer


class TimerWithLoggingCallback(Timer):
    def __init__(self, duration=timedelta(weeks=1), interval='epoch'):
        super().__init__(duration=duration, interval=interval)

        self.last_epoch_end = 0.0

    def on_train_epoch_end(self, trainer, pl_module: LightningModule):
        curr_epoch_time = self.time_elapsed('train') - self.last_epoch_end
        pl_module.log('train_time_seconds', curr_epoch_time, prog_bar=True)

        self.last_epoch_end = self.time_elapsed('train')
