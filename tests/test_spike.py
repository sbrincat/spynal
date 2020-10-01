"Tests for spike_analysis.py will go here"
import numpy as np
import pytest

from ..spike_analysis import *
from ..spike_analysis import _default_time_bins

@pytest.fixture(scope='session')
def fixture():
    data, _ = simulate_spike_trains(n_trials=20, n_conds=1, window=10.0, 
                                    seed=0)
    ndarray = data.reshape(4,5)

    return ndarray

def test_bin_count_return_shape(fixture):
    lim = [0,10]
    counts, _ = bin_count(fixture, lim=lim)
    
    assert counts.shape == (4, 5, 200)

def test_bin_count_sum(fixture):
    lim = [0,10]
    counts, _ = bin_count(fixture, lim=lim)

    assert counts.sum() == 975

def test_bin_count_bins_vs_lim(fixture):
    width = 20e-3
    lim = [0,10]

    bins = _default_time_bins(lim,width)

    counts1, _ = bin_count(fixture, width=width, lim=lim)
    counts2, _ = bin_count(fixture, width=width, bins=bins)

    assert np.all(counts1 == counts2)
