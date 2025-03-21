# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import warnings
warnings.filterwarnings("ignore")
import sys

import typer

from lightning.pytorch import Trainer

from cancer_classification.utils.mlflow_utils import get_lightning_mlflow_logger
from cancer_classification.utils.paths import get_curr_dir, get_curr_filename
from cancer_classification.utils.git_utils import check_repo_is_in_sync, commit_latest_run, GitOutOfSyncError

from cancer_classification.custom_callbacks.get_callbacks import get_callbacks
from cancer_classification.data_modules.nct_crc_he_100k__data_module import NCT_CRC_HE_100K__DataModule
from cancer_classification.lightning_modules import model_classes


app = typer.Typer()


@app.command()
def train_and_test(
    model_name: str = typer.Option(help="The name of the model to train and test."),
    experiment_name: str = typer.Option(None, "--experiment-name", help="The name of the experiment to log to.")
):
    if not experiment_name:
        experiment_name = model_name.upper()

    try:
        current_git_hash = check_repo_is_in_sync()
    except GitOutOfSyncError as e:
        sys.exit(e)

    mlflow_logger = get_lightning_mlflow_logger(experiment_name, get_curr_filename(), current_git_hash)

    trainer = Trainer(
        callbacks=get_callbacks(),
        logger=mlflow_logger,
        max_epochs=50
    )

    model = model_classes[model_name]()
    data_module = NCT_CRC_HE_100K__DataModule(logger=mlflow_logger)

    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)

    commit_latest_run(experiment_name, mlflow_logger.experiment.get_run(mlflow_logger._run_id))


if __name__ == "__main__":
    app()
