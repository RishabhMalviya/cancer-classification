# Overview
I am using this repo to build a deep learning model for classifying cancer types from on crops of WSI (Whole Slide Images). The goal is:

1. Train a deep network to classify images into one of the 9 tissue types. Run with evaluations on a validation/test dataset to get the quantitatively best model.
2. Modify the training procedures to run on with data parallelism on multiple GPUs (using [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro))
3. Export and optimize the model with ONNX
4. Run the ONNX model behind a FastAPI server

# Dataset
I am using the [NCT-CRC-HE-100K dataset](https://zenodo.org/records/1214456). It is a set of 100,000 non-overlapping image patches from hematoxylin & eosin (H&E) stained histological images of human colorectal cancer (CRC) and normal tissue. The 9 tissue classes are:
1. Adipose (ADI)
2. Background (BACK)
3. Debris (DEB)
4. Lymphocytes (LYM)
5. Mucus (MUC)
6. Smooth Muscle (MUS)
7. Normal Colon Mucosa (NORM)
8. Cancer-Associated Stroma (STR)
9. Colorectal Adenocarcinoma Epithelium (TUM)

# Experimentation
1. You can run experiments with `poetry run python ./experiments/train_and_test.py --model-name resnet18`.
2. The experiments should get tracked in MLFlow, with logs under the `experiments/logs/mlruns` directory.
3. To access the MLFlow UI, run `make mlflow` from the base directory of this repo. Currently, the best run was `nimble-crane-881` (from commit [7b0d865](https://github.com/RishabhMalviya/cancer-classification/commit/7b0d865f9c1f461eb42cdefa96006bb262b2d1e7)), with the following metrics:

| Metric          | Value       |
|------------------|-------------|
| Train Accuracy         | 99.74%       |
| Validation Accuracy           | 97.74%       |
| Test Accuracy        | 99.62%       |
| Test F1 Score        | 0.996       |
| Test Precision        | 0.995       |
| Test Recall        | 0.996       |

---
---
---

# Aside: Source Cookiecutter Template
This repo was created from my [cookie-cutter template](https://github.com/RishabhMalviya/cookiecutter-kaggle), but it has changed a lot since then. For posterity, these are the usage instructions from that repo: 

```markdown
Run `source setup.sh` to set things up.

## Usage Instructions
1. Run whatever experiment you want with a command like `python ./experiments/scripts/<experiment-name>/<main-script>`. If you've built off of the base examples in the cookiecutter project, it should save everything to MLFlow logs under the `experiments/logs/mlruns` directory.
2. To access the MLFlow UI, preferably run `mlflow ui` from the the `experiments/logs` directory. Running the command from other directories will pollute your file structure with stray `mlruns` directories.

## Directory Structure
1. `data` is meant hold the actual data and some notebooks for exploration
2. `eda` is meant to hold notebooks and generated plots from EDA
3. `experiments` is meant to hold training/evaluation scripts and notebooks.
4. The folder with the name of the project is mean to hold python modules that you want to be able to import everywhere else.
```
