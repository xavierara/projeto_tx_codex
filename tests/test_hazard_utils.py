import numpy as np

from hazard_utils import conservative_probability


def test_conservative_probability_bounds():
    probs = np.array([0.2, 0.4, 0.6])
    conservative = conservative_probability(probs, k=1.0)
    assert 0.0 <= conservative <= 1.0
