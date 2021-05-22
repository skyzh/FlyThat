#!/bin/bash

set -xe

kaggle competitions submit -c drosophila-embryos-auc -f "$1/submit_auc.csv" -m "$2 ($1)"
