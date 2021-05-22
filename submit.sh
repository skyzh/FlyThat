#!/bin/bash
set -e

echo "Submitting for AUC..."
kaggle competitions submit -c drosophila-embryos-auc -f "$1/submit_auc.csv" -m "$2 ($1)"
echo "Submitting for Samples F1..."
kaggle competitions submit -c drosophila-embryos-samples-f1 -f "$1/submit_f1.csv" -m "$2 ($1)"
echo "Submitting for Macro F1..."
kaggle competitions submit -c drosophila-embryos-macro-f1 -f "$1/submit_f1.csv" -m "$2 ($1)"
