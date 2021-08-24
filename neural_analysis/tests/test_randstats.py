""" Unit tests for randstats.py module """
import pytest
import numpy as np

from .data_fixtures import one_sample_data, two_sample_data, one_way_data, two_way_data
from ..randstats import one_sample_tstat, one_sample_test, \
                        paired_tstat, paired_sample_test, paired_sample_test_labels, \
                        two_sample_tstat, two_sample_test, two_sample_test_labels, \
                        one_way_fstat, one_way_test, two_way_fstat, two_way_test, \
                        one_sample_confints, paired_sample_confints, two_sample_confints

# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('method, result_p, result_obs, result_resmp',
                         [('randomization', 0.05, 6.91, 0.49),
                          ('bootstrap', 0.05, 6.91, 0.27)])
def test_one_sample_test(one_sample_data, method, result_p, result_obs, result_resmp):
    """ Unit tests for one_sample_test function for 1-sample randomization stats """
    data = one_sample_data
    data_orig = data.copy()

    n = int(10)
    n_chnls = int(4)
    n_resamples = int(20)

    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity
    p, stat_obs, stat_resmp = one_sample_test(data, axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (1,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples-1,n_chnls)
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with return_stats=False call
    p2 = one_sample_test(data, axis=0, method=method, seed=1,
                         n_resamples=n_resamples, return_stats=False)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert np.allclose(p, p2)

    # Test for consistent output with different data array shape (3rd axis)
    p, stat_obs, stat_resmp = one_sample_test(data.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                              axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples-1,n_chnls/2,n_chnls/2)
    assert np.isclose(p[0,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = one_sample_test(data.T, axis=-1, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (n_chnls,1)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples-1)
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = one_sample_test(data[:,0], axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert isinstance(p,float)
    assert isinstance(stat_obs,float)
    assert stat_resmp.shape == (n_resamples-1,)
    assert np.isclose(p, result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs, result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with custom stat function (but setting it equal to defaults)
    stat_func = lambda data: one_sample_tstat(data, axis=0, mu=0)
    p, stat_obs, stat_resmp = one_sample_test(data, axis=0, method=method, stat=stat_func, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('stat, method, result_p, result_obs, result_resmp',
                         [('paired', 'permutation', 0.05, -2.97, -0.35),
                          ('paired', 'bootstrap', 0.05, -2.97, -0.54),
                          ('two_sample', 'permutation', 0.05, -3.39, -0.28),
                          ('two_sample', 'bootstrap', 0.05, -3.39, -0.49)])
def test_two_sample_test(two_sample_data, stat, method, result_p, result_obs, result_resmp):
    """ Unit tests for paired_sample_test and two_sample_test functions for paired/2-sample stats """
    data, labels = two_sample_data
    data_orig = data.copy()

    data1 = data[labels == 0]
    data2 = data[labels == 1]
    data1_orig = data1.copy()
    data2_orig = data2.copy()

    test_func = paired_sample_test if stat == 'paired' else two_sample_test
    n = int(10)
    n_chnls = int(4)
    n_resamples = int(20)

    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity
    p, stat_obs, stat_resmp = test_func(data1, data2, axis=0, method=method, seed=1,
                                        n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert p.shape == (1,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples-1,n_chnls)
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with return_stats=False call
    p2 = test_func(data1, data2, axis=0, method=method, seed=1,
                   n_resamples=n_resamples, return_stats=False)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert p2.shape == p.shape
    assert np.allclose(p, p2)

    # Test for consistent output with (data,labels) version
    test_func_labels = paired_sample_test_labels if stat == 'paired' else two_sample_test_labels
    p2, stat_obs2, stat_resmp2 = test_func_labels(data, labels, axis=0, method=method, seed=1,
                                                  n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert stat_obs2.shape == stat_obs.shape
    assert stat_resmp2.shape == stat_resmp.shape
    assert np.allclose(p, p2)
    assert np.allclose(stat_obs, stat_obs2)
    assert np.allclose(stat_resmp, stat_resmp2)

    # Test for consistent output with string-valued labels
    groups = np.asarray(['cond1','cond2'])
    p2, stat_obs2, stat_resmp2 = test_func_labels(data, groups[labels], axis=0, method=method, seed=1,
                                                  n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert stat_obs2.shape == stat_obs.shape
    assert stat_resmp2.shape == stat_resmp.shape
    assert np.allclose(p, p2)
    assert np.allclose(stat_obs, stat_obs2)
    assert np.allclose(stat_resmp, stat_resmp2)

    # Test for consistent output using groups argument to subset data
    p2, stat_obs2, stat_resmp2 = test_func_labels(np.concatenate((data,data),axis=0),
                                                  np.hstack((labels,labels+2)), groups=[2,3],
                                                  axis=0, method=method, seed=1,
                                                  n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert stat_obs2.shape == stat_obs.shape
    assert stat_resmp2.shape == stat_resmp.shape
    assert np.allclose(p, p2)
    assert np.allclose(stat_obs, stat_obs2)
    assert np.allclose(stat_resmp, stat_resmp2)

    # Test for consistent output with different data array shape (3rd axis)
    p, stat_obs, stat_resmp = test_func(data1.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                        data2.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                        axis=0, method=method, seed=1,
                                        n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert p.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples-1,n_chnls/2,n_chnls/2)
    assert np.isclose(p[0,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = test_func(data1.T, data2.T, axis=-1, method=method, seed=1,
                                        n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert p.shape == (n_chnls,1)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples-1)
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = test_func(data1[:,0], data2[:,0], axis=0, method=method, seed=1,
                                        n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert isinstance(p,float)
    assert isinstance(stat_obs,float)
    assert stat_resmp.shape == (n_resamples-1,)
    assert np.isclose(p, result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs, result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with custom stat function (but setting it equal to defaults)
    if stat == 'paired':    stat_func = lambda data1,data2: paired_tstat(data1, data2, axis=0, d=0)
    else:                   stat_func = lambda data1,data2: two_sample_tstat(data1, data2, axis=0, d=0)
    # TEMP HACK Skip this for paired tests
    if stat == 'two_sample':
        p, stat_obs, stat_resmp = test_func(data1, data2, axis=0, method=method, stat=stat_func, seed=1,
                                            n_resamples=n_resamples, return_stats=True)
        assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
        assert np.array_equal(data2,data2_orig)
        assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
        assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
        assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('method, result_p, result_obs, result_resmp',
                         [('permutation', 0.05, 11.50, 0.41)])
def test_one_way_test(one_way_data, method, result_p, result_obs, result_resmp):
    """ Unit tests for one_way_test function for ANOVA-like 1-way stats """
    data, labels = one_way_data
    data_orig = data.copy()

    n = 10
    n_groups = 3
    n_chnls = 4
    n_resamples = 20

    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity
    p, stat_obs, stat_resmp = one_way_test(data, labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (1,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples-1,n_chnls)
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with return_stats=False call
    p2 = one_way_test(data, labels, axis=0, method=method, seed=1,
                      n_resamples=n_resamples, return_stats=False)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert np.allclose(p, p2)

    # Test for consistent output with string-valued labels
    groups = np.asarray(['cond1','cond2','cond3'])
    p2, stat_obs2, stat_resmp2 = one_way_test(data, groups[labels], axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert stat_obs2.shape == stat_obs.shape
    assert stat_resmp2.shape == stat_resmp.shape
    assert np.allclose(p, p2)
    assert np.allclose(stat_obs, stat_obs2)
    assert np.allclose(stat_resmp, stat_resmp2)

    # Test for consistent output using groups argument to subset data
    p2, stat_obs2, stat_resmp2 = one_way_test(np.concatenate((data,data),axis=0),
                                              np.hstack((labels,labels+3)), groups=[3,4,5],
                                              axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert stat_obs2.shape == stat_obs.shape
    assert stat_resmp2.shape == stat_resmp.shape
    assert np.allclose(p, p2)
    assert np.allclose(stat_obs, stat_obs2)
    assert np.allclose(stat_resmp, stat_resmp2)

    # Test for consistent output with different data array shape (3rd axis)
    p, stat_obs, stat_resmp = one_way_test(data.reshape((n*n_groups,int(n_chnls/2),int(n_chnls/2))),
                                           labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples-1,n_chnls/2,n_chnls/2)
    assert np.isclose(p[0,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = one_way_test(data.T, labels, axis=-1, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (n_chnls,1)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples-1)
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = one_way_test(data[:,0], labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert isinstance(p,float)
    assert isinstance(stat_obs,float)
    assert stat_resmp.shape == (n_resamples-1,)
    assert np.isclose(p, result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs, result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with custom stat function (but setting it equal to defaults)
    stat_func = lambda data,labels: one_way_fstat(data, labels, axis=0)
    p, stat_obs, stat_resmp = one_way_test(data, labels, axis=0, method=method, stat=stat_func,
                                           seed=1, n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('method, result_p, result_obs, result_resmp',
                         [('permutation', (0.05,0.05,0.4), (3.10,24.91,0.18), (0.25,0.19,0.20))])
def test_two_way_test(two_way_data, method, result_p, result_obs, result_resmp):
    """ Unit tests for two_way_test function for ANOVA-like 2-way stats """
    data, labels = two_way_data
    data_orig = data.copy()

    n = 10
    n_groups = 4
    n_terms = 3
    n_chnls = 4
    n_resamples = 20

    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity
    p, stat_obs, stat_resmp = two_way_test(data, labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    # print(np.round(p[:,0],2), np.round(stat_obs[:,0],2), np.round(stat_resmp[:,0,:].mean(axis=-1),2))
    assert p.shape == (n_terms,n_chnls)
    assert stat_obs.shape == (n_terms,n_chnls)
    assert stat_resmp.shape == (n_terms,n_chnls,n_resamples-1)
    assert np.allclose(p[:,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[:,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[:,0,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with return_stats=False call
    p2 = two_way_test(data, labels, axis=0, method=method, seed=1,
                      n_resamples=n_resamples, return_stats=False)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert np.allclose(p, p2)

    # Test for consistent output with string-valued labels
    groups = np.asarray(['cond1','cond2','cond3','cond4'])
    p2, stat_obs2, stat_resmp2 = two_way_test(data, groups[labels], axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p2.shape == p.shape
    assert stat_obs2.shape == stat_obs.shape
    assert stat_resmp2.shape == stat_resmp.shape
    assert np.allclose(p, p2)
    assert np.allclose(stat_obs, stat_obs2)
    assert np.allclose(stat_resmp, stat_resmp2)

    # Test for consistent output with different data array shape (3rd axis)
    p, stat_obs, stat_resmp = two_way_test(data.reshape((n*n_groups,int(n_chnls/2),int(n_chnls/2))),
                                           labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (n_terms,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (n_terms,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_terms,n_chnls/2,n_chnls/2,n_resamples-1)
    assert np.allclose(p[:,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[:,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[:,0,0,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = two_way_test(data.T, labels, axis=-1, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (n_chnls,n_terms)
    assert stat_obs.shape == (n_chnls,n_terms)
    assert stat_resmp.shape == (n_chnls,n_terms,n_resamples-1)
    assert np.allclose(p[0,:], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[0,:], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[0,:,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = two_way_test(data[:,0], labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (n_terms,)
    assert stat_obs.shape == (n_terms,)
    assert stat_resmp.shape == (n_terms,n_resamples-1)
    assert np.allclose(p[:], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[:], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[:,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for expected output shape without interaction term call
    p, stat_obs, stat_resmp = two_way_test(data, labels[:,:-1], axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert p.shape == (n_terms-1,n_chnls)
    assert stat_obs.shape == (n_terms-1,n_chnls)
    assert stat_resmp.shape == (n_terms-1,n_chnls,n_resamples-1)

    # Test for consistent output with custom stat function (but setting it equal to defaults)
    stat_func = lambda data,labels: two_way_fstat(data, labels, axis=0)
    p, stat_obs, stat_resmp = two_way_test(data, labels, axis=0, method=method, stat=stat_func,
                                           seed=1, n_resamples=n_resamples, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.allclose(p[:,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[:,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[:,0,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('method, result_ci, result_obs, result_resmp',
                         [('bootstrap', (8.53,12.18), 10.35, 10.36)])
def test_one_sample_confints(one_sample_data, method, result_ci, result_obs, result_resmp):
    """ Unit tests for one_sample_confints function for 1-sample confidence intervals """
    data = one_sample_data
    data_orig = data.copy()

    n = int(10)
    n_chnls = int(4)
    n_resamples = int(40)

    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity
    ci, stat_obs, stat_resmp = one_sample_confints(data, axis=0, n_resamples=n_resamples,
                                                   seed=1, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert ci.shape == (2,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples,n_chnls)
    print(ci.shape, stat_obs.shape, stat_resmp.shape)
    print(ci[:,0], stat_obs[0,0], stat_resmp[:,0].mean())
    assert np.allclose(ci[:,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with return_stats=False call
    ci2 = one_sample_confints(data, axis=0, n_resamples=n_resamples, seed=1, return_stats=False)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert ci2.shape == ci.shape
    assert np.allclose(ci, ci2)

    # Test for consistent output with different data array shape (3rd axis)
    ci, stat_obs, stat_resmp = one_sample_confints(data.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                                   axis=0, n_resamples=n_resamples,
                                                   seed=1, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert ci.shape == (2,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples,n_chnls/2,n_chnls/2)
    assert np.allclose(ci[:,0,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    ci, stat_obs, stat_resmp = one_sample_confints(data.T, axis=-1, n_resamples=n_resamples,
                                                   seed=1, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert ci.shape == (n_chnls,2)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples)
    assert np.allclose(ci[0,:], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    ci, stat_obs, stat_resmp = one_sample_confints(data[:,0], axis=0, n_resamples=n_resamples,
                                                   seed=1, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert ci.shape == (2,)
    assert isinstance(stat_obs,float)
    assert stat_resmp.shape == (n_resamples,)
    assert np.allclose(ci, result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs, result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with custom stat function (but setting it equal to defaults)
    stat_func = lambda data: np.mean(data, axis=0, keepdims=True)
    ci, stat_obs, stat_resmp = one_sample_confints(data, axis=0, stat=stat_func, n_resamples=n_resamples,
                                                  seed=1, return_stats=True)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.allclose(ci[:,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('stat, method, result_ci, result_obs, result_resmp',
                         [('paired', 'bootstrap', (-14.55,-4.77), -9.10, -9.84),
                          ('two_sample', 'bootstrap', (-14.58,-5.40), -9.10, -9.05)])
def test_two_sample_confints(two_sample_data, stat, method, result_ci, result_obs, result_resmp):
    """ Unit tests for paired/two_sample_confints function for paired/two-sample CI's """
    data, labels = two_sample_data
    data1 = data[labels == 0]
    data2 = data[labels == 1]
    data1_orig = data1.copy()
    data2_orig = data2.copy()

    test_func = paired_sample_confints if stat == 'paired' else two_sample_confints
    n = int(10)
    n_chnls = int(4)
    n_resamples = int(40)

    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity
    ci, stat_obs, stat_resmp = test_func(data1, data2, axis=0, n_resamples=n_resamples,
                                         seed=1, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert ci.shape == (2,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples,n_chnls)
    print(ci.shape, stat_obs.shape, stat_resmp.shape)
    print(ci[:,0], stat_obs[0,0], stat_resmp[:,0].mean())
    assert np.allclose(ci[:,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with return_stats=False call
    ci2 = test_func(data1, data2, axis=0, n_resamples=n_resamples, seed=1, return_stats=False)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert ci2.shape == ci.shape
    assert np.allclose(ci, ci2)

    # Test for consistent output with different data array shape (3rd axis)
    ci, stat_obs, stat_resmp = test_func(data1.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                         data2.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                         axis=0, n_resamples=n_resamples,
                                         seed=1, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert ci.shape == (2,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples,n_chnls/2,n_chnls/2)
    assert np.allclose(ci[:,0,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    ci, stat_obs, stat_resmp = test_func(data1.T, data2.T, axis=-1, n_resamples=n_resamples,
                                                   seed=1, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert ci.shape == (n_chnls,2)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples)
    assert np.allclose(ci[0,:], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    ci, stat_obs, stat_resmp = test_func(data1[:,0], data2[:,0], axis=0, n_resamples=n_resamples,
                                         seed=1, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert ci.shape == (2,)
    assert isinstance(stat_obs,float)
    assert stat_resmp.shape == (n_resamples,)
    assert np.allclose(ci, result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs, result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:].mean(), result_resmp, rtol=1e-2, atol=1e-2)

    # Test for consistent output with custom stat function (but setting it equal to defaults)
    if stat == 'paired':
        # TEMP Until we clear up paired function situation
        stat_func = lambda diff: np.mean(diff, axis=0, keepdims=True)
        # stat_func = lambda data1,data2: np.mean(data1 - data2, axis=0, keepdims=True)
    else:
        stat_func = lambda data1,data2: np.mean(data1, axis=0, keepdims=True) - \
                                        np.mean(data2, axis=0, keepdims=True)
    ci, stat_obs, stat_resmp = test_func(data1, data2, axis=0, stat=stat_func, n_resamples=n_resamples,
                                         seed=1, return_stats=True)
    assert np.array_equal(data1,data1_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data2,data2_orig)
    assert np.allclose(ci[:,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
