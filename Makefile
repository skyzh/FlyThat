MODEL_PATH = "fly_bitch/model.pkl"
TENSORBOARD_LOGS_PATH = "fly_bitch/runs/logs"

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

run_train:
	python -m fly_bitch train $(TENSORBOARD_LOGS_PATH) $(MODEL_PATH)

