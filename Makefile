SHELL := /bin/bash

notebook:
	poetry run jupyter notebook --no-browser --port 8080

mlflow:
	cd ./experiments/logs && poetry run mlflow ui

train_resnet:
	poetry run python ./experiments/train_and_test.py --model-name resnet18
