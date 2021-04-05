import pytest
from ..dataset import gen_onehot


def test_onehot_gen():
    onehot_vec = gen_onehot([1, 3, 5], 10)
    assert onehot_vec.tolist() == [0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
