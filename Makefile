SHELL := /bin/bash

notebook:
	poetry run jupyter notebook --no-browser --port 8080

mlflow:
	cd ./experiments/logs && poetry run mlflow ui

train_resnet:
	poetry run python ./experiments/scripts/resnet/train_and_test.py train_and_test --model_class resnet18
