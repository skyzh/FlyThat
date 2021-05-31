import pytest
from ..train import result_threshold
import numpy as np


def test_result_threshold():
    result = result_threshold(np.array([[0.0, 1.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.1, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.1, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.1, 0.0]]), True)
    assert np.all(
        result == np.array([[0, 1, 0, 1, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0]]))
