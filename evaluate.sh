#!/bin/bash

set -e

DATA_PATH=$(pwd)/data

python -m fly_bitch evaluate --data ${DATA_PATH} $@
