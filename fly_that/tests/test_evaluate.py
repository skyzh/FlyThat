import pytest
from ..evaluate import parse_file_name


def test_parse_filename():
    epoch, auc, f1macro, f1micro = parse_file_name(
        "model_6_0.5514872642167922_0.143187346708827_0.3008971704623879.pkl")
    assert epoch == 6
    assert auc == 0.5514872642167922
    assert f1macro == 0.143187346708827
    assert f1micro == 0.3008971704623879
