""" Unit tests for utils.py module """
import pytest
import random
import numpy as np

from neural_analysis.tests.data_fixtures import one_sample_data, two_sample_data, two_way_data, \
                                                MISSING_ARG_ERRS
from neural_analysis.utils import zscore, one_sample_tstat, paired_tstat, two_sample_tstat, \
                                  one_way_fstat, two_way_fstat, fano, cv, cv2, lv, \
                                  correlation, rank_correlation, set_random_seed, \
                                  gaussian, gaussian_2d, gaussian_nd


# =============================================================================
# Unit tests for statistics functions
# =============================================================================
def test_zscore(one_sample_data):
    """ Unit tests for zscore() function """
    data = one_sample_data
    data_orig = data.copy()
    n_trials,n_chnls = data.shape

    # Basic test of shape, value of output. Test that z-scored data has mean ~ 0, sd ~ 1
    z = zscore(data[:,0])
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert z.shape == (n_trials,)
    assert np.isclose(z.mean(), 0, rtol=1e-2, atol=1e-2)
    assert np.isclose(z.std(), 1, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims
    z = zscore(data, axis=0)
    assert z.shape == (n_trials,n_chnls)
    assert np.allclose(z.mean(axis=0), np.zeros((1,n_chnls)), rtol=1e-2, atol=1e-2)
    assert np.allclose(z.std(axis=0), np.ones((1,n_chnls)), rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    z = zscore(data.T, axis=1)
    assert z.shape == (n_chnls,n_trials)
    assert np.allclose(z.mean(axis=1), np.zeros((n_chnls,1)), rtol=1e-2, atol=1e-2)
    assert np.allclose(z.std(axis=1), np.ones((n_chnls,1)), rtol=1e-2, atol=1e-2)

    # Test for proper function with zscore across entire array
    z = zscore(data, axis=None)
    assert z.shape == (n_trials,n_chnls)
    assert np.isclose(z.mean(), 0, rtol=1e-2, atol=1e-2)
    assert np.isclose(z.std(), 1, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        z = zscore(data, foo=None)


@pytest.mark.parametrize('stat_func,    result',
                         [(fano,        1.95),
                          (cv,          0.43),
                          (cv2,         0.57),
                          (lv,          0.31)])
def test_snr_stats(one_sample_data, stat_func, result):
    """ Unit tests for signal-to-noise-type stats (Fano, CV, etc.) """
    data = one_sample_data
    data_orig = data.copy()
    n_datapoints, n_chnls = data.shape

    # Basic test of shape, value of output
    stat = stat_func(data[:,0], axis=0)
    print(np.round(stat,2))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.isscalar(stat)
    assert np.isclose(stat, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims
    stat = stat_func(data[:,:], axis=0)
    assert stat.shape == (1,n_chnls)
    assert np.isclose(stat[:,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    stat = stat_func(data[:,:].T, axis=1)
    assert stat.shape == (n_chnls,1)
    assert np.isclose(stat[0,:], result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        stat = stat_func(data[:,0], axis=0, foo=None)


@pytest.mark.parametrize('stat_func,    result',
                         [(one_sample_tstat,    6.91),
                          (paired_tstat,        -2.97),
                          (two_sample_tstat,    -3.39),
                          (one_way_fstat,        1.79),
                          (two_way_fstat,       (3.10,24.91,0.18))])
def test_parametric_stats(two_way_data, stat_func, result):
    data, labels = two_way_data
    if stat_func is one_sample_tstat:
        data = data[:10,:]
    elif stat_func in [paired_tstat,two_sample_tstat]:
        data = data[:20,:]
        labels = labels[:20,0]
    elif stat_func is one_way_fstat:
        labels = labels[:,0]

    data_orig = data.copy()
    labels_orig = labels.copy()
    n_datapoints, n_chnls = data.shape

    extra_args = dict(axis=0)
    if stat_func is not two_way_fstat: extra_args.update(keepdims=True)

    # Basic test of shape, value of output
    if stat_func is one_sample_tstat:
        stat = stat_func(data[:,0], **extra_args)
    elif stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==0,0], data[labels==1,0], **extra_args)
    else:
        stat = stat_func(data[:,0], labels, **extra_args)
    print(np.round(stat,2))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(labels,labels_orig)
    if stat_func is two_way_fstat:
        assert stat.shape == (3,)
        assert np.allclose(stat, result, rtol=1e-2, atol=1e-2)
    else:
        assert np.isscalar(stat)
        assert np.isclose(stat, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims = True
    if stat_func is one_sample_tstat:
        stat = stat_func(data[:,:], **extra_args)
    elif stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==0,:], data[labels==1,:], **extra_args)
    else:
        stat = stat_func(data, labels, **extra_args)
    if stat_func is two_way_fstat:
        assert stat.shape == (3,n_chnls)
        assert np.allclose(stat[:,0], result, rtol=1e-2, atol=1e-2)
    else:
        assert stat.shape == (1,n_chnls)
        assert np.isclose(stat[0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims = False
    if stat_func is not two_way_fstat:
        if stat_func is one_sample_tstat:
            stat = stat_func(data[:,:], axis=0, keepdims=False)
        elif stat_func in [paired_tstat,two_sample_tstat]:
            stat = stat_func(data[labels==0,:], data[labels==1,:], axis=0, keepdims=False)
        else:
            stat = stat_func(data, labels, axis=0, keepdims=False)
        assert stat.shape == (n_chnls,)
        assert np.isclose(stat[0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    extra_args['axis'] = 1
    if stat_func is one_sample_tstat:
        stat = stat_func(data[:,:].T, **extra_args)
    elif stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==0,:].T, data[labels==1,:].T, **extra_args)
    else:
        stat = stat_func(data.T, labels, **extra_args)
    if stat_func is two_way_fstat:
        assert stat.shape == (n_chnls,3)
        assert np.allclose(stat[0,:], result, rtol=1e-2, atol=1e-2)
    else:
        assert stat.shape == (n_chnls,1)
        assert np.isclose(stat[0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with arguments swapped.
    # Stat should have same magnitude, opposite sign for 2-sample stats.
    if stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==1,0], data[labels==0,0], axis=0)
        assert np.isscalar(stat)
        assert np.isclose(stat, -result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        if stat_func is one_sample_tstat:
            stat = stat_func(data[:,0], axis=0, foo=None)
        elif stat_func in [paired_tstat,two_sample_tstat]:
            stat = stat_func(data[labels==0,0], data[labels==1,0], axis=0, foo=None)
        else:
            stat = stat_func(data[:,0], labels, axis=0, foo=None)


@pytest.mark.parametrize('corr_type, result, result2',
                         [('pearson',   -0.33, -0.20),
                          ('spearman',  -0.36, -0.26)])
def test_correlation(two_sample_data, corr_type, result, result2):
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

    # Test for proper function with correlation across entire array
    r = corr_func(data[labels==0,:], data[labels==1,:], axis=None)
    assert np.isscalar(r)
    assert np.isclose(r, result2, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        r = corr_func(data[labels==0,:], data[labels==1,:], foo=None)


# =============================================================================
# Unit tests for numerical functions
# =============================================================================
@pytest.mark.parametrize('rand_func', [np.random.rand, random.random])
def test_set_random_seed(rand_func):
    """ Unit tests for test_random_seed() """
    # Generate unseeded random number using given random number generator
    unseeded = rand_func()

    # Seed random number generators and generate seeded random number
    seed = set_random_seed()
    seeded = rand_func()

    # Generate seeded random number with same seed
    set_random_seed(seed)
    seeded2 = rand_func()

    # Check that both seeded values are identical and unseeded value is not
    assert seeded == seeded2
    assert seeded != unseeded


def test_gaussian(one_sample_data):
    """ Unit tests for gaussian function """
    data = (one_sample_data[:,0] - 10)/5
    data_orig = data.copy()
    n = data.shape[0]
    result = 0.98

    # Basic test of shape, value of output    
    f_x = gaussian(data)
    print(np.round(f_x[0],2))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert f_x.shape == (n,)
    assert np.isclose(f_x[0], result, rtol=1e-2, atol=1e-2)
    
    # Test for consistent results with scalar input
    f_x = gaussian(data[0])
    assert np.isscalar(f_x)  
    assert np.isclose(f_x, result, rtol=1e-2, atol=1e-2)
        
    # Test for consistent results with hand-set parameters 
    f_x = gaussian(data, center=0, width=1, amplitude=1, baseline=0)
    assert np.isclose(f_x[0], result, rtol=1e-2, atol=1e-2)
    
    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        f_x = gaussian(data, foo=None)
    