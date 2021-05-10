#!/bin/bash

set -e

MODEL_PATH=$(pwd)/runs/model
TENSORBOARD_LOGS_PATH=$(pwd)/runs/logs
DATA_PATH=$(pwd)/data

python -m fly_bitch train --log ${TENSORBOARD_LOGS_PATH} --model ${MODEL_PATH} --data ${DATA_PATH}
