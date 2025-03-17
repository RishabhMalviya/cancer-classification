# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import shutil


from datetime import timedelta

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    RichProgressBar, EarlyStopping
)

from cancer_classification.utils.mlflow_utils import get_lightning_mlflow_logger, ModelCheckpointWithCleanupCallback
from cancer_classification.utils.paths import get_curr_dir, get_curr_filename
from cancer_classification.utils.git_utils import check_repo_is_in_sync, commit_latest_run, GitOutOfSyncError

from cancer_classification.custom_callbacks.timer_with_logging_callback import TimerWithLoggingCallback

from cancer_classification.data_modules.nct_crc_he_100k__data_module import NCT_CRC_HE_100K__DataModule

from experiments.scripts.resnet.model import ResNet18__LightningModule


EXPERIMENT_NAME = get_curr_dir().upper()


def _configure_callbacks():
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode='min',
        patience=10,
        stopping_threshold=0.05,
        divergence_threshold=5.0
    )

    checkpoint_callback = ModelCheckpointWithCleanupCallback(
        save_top_k=2,
        save_last=True,
        monitor="val_loss",
        mode="min",
        verbose=True,
        filename='{epoch}-{val_loss:.2f}'
    )

    timer_with_logging_callback = TimerWithLoggingCallback(duration=timedelta(weeks=1), interval='epoch')

    return [
        early_stopping_callback,
        checkpoint_callback,
        timer_with_logging_callback,
        RichProgressBar()
    ]


def cli_main(_mlflow_logger):
    model = ResNet18__LightningModule()
    data_module = NCT_CRC_HE_100K__DataModule()

    trainer = Trainer(
        callbacks=_configure_callbacks(),
        logger=_mlflow_logger,
        max_epochs=1
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    try:
        current_git_hash = check_repo_is_in_sync()
    except GitOutOfSyncError as e:
        sys.exit(e)

    try:
        mlflow_logger = get_lightning_mlflow_logger(EXPERIMENT_NAME, get_curr_filename(), current_git_hash)
        cli_main(mlflow_logger)
        commit_latest_run(EXPERIMENT_NAME, mlflow_logger.experiment.get_run(mlflow_logger._run_id))
    finally:
        local_artifacts_dir = os.path.join(get_curr_dir(), '../../../', mlflow_logger.experiment_id)
        if os.path.exists(local_artifacts_dir):
            shutil.rmtree(local_artifacts_dir)

