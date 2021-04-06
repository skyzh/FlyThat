test:
	pytest fly_bitch

format:
	autopep8 --in-place --recursive fly_bitch

run_data_parser:
	python -m fly_bitch data_parser

run_dataset:
	python -m fly_bitch dataset

run_model:
	python -m fly_bitch model

run_train:
	python -m fly_bitch train
