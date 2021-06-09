#!/bin/bash

set -e

DATA_PATH=$(pwd)/data

python -m fly_that evaluate --data ${DATA_PATH} $@
