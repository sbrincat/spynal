""" Unit tests for utils.py module """
import pytest
import numpy as np

from neural_analysis.tests.data_fixtures import two_sample_data
from neural_analysis.utils import correlation, rank_correlation


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('corr_type, result',
                         [('pearson',   -0.33),
                          ('spearman',  -0.36)])
def test_correlation(two_sample_data, corr_type, result):
    """ Unit tests for correlation() and rank_correlation() functions """
    data, labels = two_sample_data
    data_orig = data.copy()

    corr_func = correlation if corr_type == 'pearson' else rank_correlation

    # Basic test of shape, value of output
    r = corr_func(data[labels==0,0], data[labels==1,0], keepdims=False)
    print(np.round(r,2), type(r))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.isscalar(r)
    assert np.isclose(r, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with arguments swapped
    r = corr_func(data[labels==1,0], data[labels==0,0], keepdims=False)
    assert np.isclose(r, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims
    r = corr_func(data[labels==0,:], data[labels==1,:], axis=0, keepdims=True)
    assert r.shape == (1,4)
    assert np.isclose(r[:,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    r = corr_func(data[labels==0,:].T, data[labels==1,:].T, axis=1, keepdims=True)
    assert r.shape == (4,1)
    assert np.isclose(r[0,:], result, rtol=1e-2, atol=1e-2)
