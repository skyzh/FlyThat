MODEL_PATH = "runs/model"
TENSORBOARD_LOGS_PATH = "runs/logs"

test:
	pytest fly_that -vvv

format:
	autopep8 --in-place --recursive fly_that

tensorboard:
	tensorboard --logdir=$(TENSORBOARD_LOGS_PATH)

run_data_parser:
	python -m fly_that data_parser

run_dataset:
	python -m fly_that dataset

run_model:
	python -m fly_that model

run_loss:
	python -m fly_that loss

clean:
	rm -rf $(shell pwd)/runs
