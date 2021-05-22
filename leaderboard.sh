#!/bin/bash
set -e

TEAM_NAME="very-trivial-baseline"

kaggle competitions leaderboard -c drosophila-embryos-auc --show | grep ${TEAM_NAME} -C 10 --color
kaggle competitions leaderboard -c drosophila-embryos-samples-f1 --show | grep ${TEAM_NAME} -C 10 --color
kaggle competitions leaderboard -c drosophila-embryos-macro-f1 --show | grep ${TEAM_NAME} -C 10 --color
