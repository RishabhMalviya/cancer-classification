import os
import warnings
warnings.filterwarnings("ignore")
import logging

import mlflow
from mlflow.exceptions import MlflowException

import matplotlib.pyplot as plt
import seaborn as sns

from cancer_classification.utils.paths import get_curr_dir, EXPERIMENT_LOGS_DIR


EXPERIMENT_NAME = get_curr_dir().upper()

mlflow.set_tracking_uri(os.path.join(EXPERIMENT_LOGS_DIR, './mlruns'))

experiment_name = 'MLFLOW_TEST'

try:
    mlflow.create_experiment(
        experiment_name,
    )
except MlflowException:
    logging.info(f'Experiment {experiment_name} already exists.')
finally:
    mlflow.set_experiment(experiment_name)


if __name__ == "__main__":
    with mlflow.start_run(tags=None):
        data = sns.load_dataset('iris')
        
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='sepal_length', y='sepal_width', hue='species')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        plt.close()

        mlflow.log_figure(fig, "iris_scatter_plot.png")
