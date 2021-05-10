MODEL_PATH = "runs/model"
TENSORBOARD_LOGS_PATH = "runs/logs"

test:
	pytest fly_bitch

format:
	autopep8 --in-place --recursive fly_bitch

tensorboard:
	tensorboard --logdir=$(TENSORBOARD_LOGS_PATH)

run_data_parser:
	python -m fly_bitch data_parser

run_dataset:
	python -m fly_bitch dataset

run_model:
	python -m fly_bitch model

