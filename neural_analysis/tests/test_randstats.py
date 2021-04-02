""" Unit tests for randstats.py module """
import pytest
import numpy as np

from scipy.stats import norm

from ..randstats import one_sample_tstat, one_sample_test, \
                        paired_tstat, paired_sample_test, paired_sample_test_labels, \
                        two_sample_tstat, two_sample_test, two_sample_test_labels, \
                        one_way_fstat, one_way_test, two_way_fstat, two_way_test, \
                        bootstrap_confints

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def one_sample_data():
    """ 
    Fixture simulates one-sample (single-condition) random data to use for unit tests 
    
    RETURNS
    data    (10,4) ndarray. Simulated data, simulating (10 trials x 4 channels)
    """    
    # Note: seed=1 makes data reproducibly match output of Matlab    
    np.random.seed(1)
    
    n       = 10
    n_chnls = 4
    mu      = 10.0
    sd      = 5.0
    return norm.ppf(np.random.rand(n,n_chnls), loc=mu, scale=sd)


@pytest.fixture(scope='session')
def two_sample_data():
    """ 
    Fixture simulates set of two-sample (2 condition) random data to use for unit tests 
    
    RETURNS
    data    (20,4) ndarray. Simulated data, simulating (10 trials*2 conditions/levels x 4 channels)
    
    labels  (20,) ndarray of int8. Set of condition labels corresponding to each trial in data
    """    
    # Note: seed=1 makes data reproducibly match output of Matlab    
    np.random.seed(1)
    
    n       = 10
    n_chnls = 4
    mu      = (10.0, 20.0)
    sd      = 5.0
    data = np.concatenate((norm.ppf(np.random.rand(n,n_chnls), loc=mu[0], scale=sd),
                           norm.ppf(np.random.rand(n,n_chnls), loc=mu[1], scale=sd)), axis=0)
    labels = np.hstack((np.zeros((n,),dtype='uint8'), np.ones((n,),dtype='uint8')))
    
    return data, labels


@pytest.fixture(scope='session')
def one_way_data():
    """ 
    Fixture simulates set of 3-condition random data to use for unit tests of one-way test functions
    
    RETURNS
    data    (30,4) ndarray. Simulated data, simulating (10 trials*3 conditions x 4 channels)
    
    labels  (30,) ndarray of int8. Set of condition labels corresponding to each trial in data
    """    
    # Note: seed=1 makes data reproducibly match output of Matlab    
    np.random.seed(1)
    
    n       = 10
    n_groups= 3
    n_chnls = 4
    mu      = (10.0, 20.0, 30.0)
    sd      = 5.0
    data = np.concatenate([norm.ppf(np.random.rand(n,n_chnls), loc=mu[i], scale=sd) 
                           for i in range(n_groups)], axis=0)
    labels = np.hstack([i*np.ones((n,),dtype='uint8') for i in range(3)])
    
    return data, labels


@pytest.fixture(scope='session')
def two_way_data():
    """ 
    Fixture simulates set of 4-condition random data to use for unit tests of two-way test functions
    
    RETURNS
    data    (40,4) ndarray. Simulated data, simulating (10 trials*4 conditions/levels x 4 channels)
    
    labels  (40,2) ndarray of int8. Set of condition labels corresponding to each trial in data
    """    
    # Note: seed=1 makes data reproducibly match output of Matlab    
    np.random.seed(1)
    
    n       = 10
    n_groups= 4
    n_terms = 3
    n_chnls = 4
    mu      = (10.0, 20.0, 30.0, 40.0)
    sd      = 5.0
    data = np.concatenate([norm.ppf(np.random.rand(n,n_chnls), loc=mu[i], scale=sd) 
                           for i in range(n_groups)], axis=0)
    labels = np.empty((n*n_groups,n_terms), dtype='uint8')
    labels[:,0] = np.tile(np.hstack((np.zeros((n,)), np.ones((n,)))), (2,))
    labels[:,1] = np.hstack((np.zeros((n*2,)), np.ones((n*2,))))
    labels[:,2] = np.hstack([i*np.ones((n,)) for i in range(n_groups)])
            
    return data, labels


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('method, result_p, result_obs, result_resmp',
                         [('randomization', 0.05, 6.91, 0.49),
                          ('bootstrap', 0.05, 6.91, 0.27)])                        
def test_one_sample_test(one_sample_data, method, result_p, result_obs, result_resmp):    
    """ Unit tests for one_sample_test function for 1-sample randomization stats """    
    data = one_sample_data
    
    n = int(10)
    n_chnls = int(4)
    n_resamples = int(20)
    
    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity    
    p, stat_obs, stat_resmp = one_sample_test(data, axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert p.shape == (1,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples-1,n_chnls)    
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with return_stats=False call
    p2 = one_sample_test(data, axis=0, method=method, seed=1,
                         n_resamples=n_resamples, return_stats=False)
    assert p2.shape == p.shape
    assert np.allclose(p, p2)
        
    # Test for consistent output with different data array shape (3rd axis)
    p, stat_obs, stat_resmp = one_sample_test(data.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                              axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert p.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples-1,n_chnls/2,n_chnls/2)    
    assert np.isclose(p[0,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
                
    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = one_sample_test(data.T, axis=-1, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert p.shape == (n_chnls,1)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples-1)    
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = one_sample_test(data[:,0], axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
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
    data1 = data[labels == 0]
    data2 = data[labels == 1]
    
    test_func = paired_sample_test if stat == 'paired' else two_sample_test
    n = int(10)
    n_chnls = int(4)
    n_resamples = int(20)
    
    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity    
    p, stat_obs, stat_resmp = test_func(data1, data2, axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
    assert p.shape == (1,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples-1,n_chnls)    
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with return_stats=False call
    p2 = test_func(data1, data2, axis=0, method=method, seed=1,
                   n_resamples=n_resamples, return_stats=False)
    assert p2.shape == p.shape
    assert np.allclose(p, p2)
        
    # Test for consistent output with (data,labels) version
    test_func_labels = paired_sample_test_labels if stat == 'paired' else two_sample_test_labels
    p2, stat_obs2, stat_resmp2 = test_func_labels(data, labels, axis=0, method=method, seed=1,
                                                  n_resamples=n_resamples, return_stats=True)
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
    assert p.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples-1,n_chnls/2,n_chnls/2)    
    assert np.isclose(p[0,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
                
    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = test_func(data1.T, data2.T, axis=-1, method=method, seed=1,
                                        n_resamples=n_resamples, return_stats=True)
    assert p.shape == (n_chnls,1)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples-1)    
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = test_func(data1[:,0], data2[:,0], axis=0, method=method, seed=1,
                                        n_resamples=n_resamples, return_stats=True)
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
        assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
        assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
        assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
                                                    
        
@pytest.mark.parametrize('method, result_p, result_obs, result_resmp',
                         [('permutation', 0.05, 11.50, 0.41)])                        
def test_one_way_test(one_way_data, method, result_p, result_obs, result_resmp):    
    """ Unit tests for one_way_test function for ANOVA-like 1-way stats """
    data, labels = one_way_data
    
    n = 10
    n_groups = 3
    n_chnls = 4
    n_resamples = 20
    
    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity    
    p, stat_obs, stat_resmp = one_way_test(data, labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    # print(p[0,0], stat_obs[0,0], stat_resmp[:,0].mean())
    assert p.shape == (1,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples-1,n_chnls)    
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with return_stats=False call
    p2 = one_way_test(data, labels, axis=0, method=method, seed=1,
                      n_resamples=n_resamples, return_stats=False)
    assert p2.shape == p.shape
    assert np.allclose(p, p2)        
        
    # Test for consistent output with string-valued labels                                        
    groups = np.asarray(['cond1','cond2','cond3'])
    p2, stat_obs2, stat_resmp2 = one_way_test(data, groups[labels], axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
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
    assert p.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples-1,n_chnls/2,n_chnls/2)    
    assert np.isclose(p[0,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
                
    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = one_way_test(data.T, labels, axis=-1, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert p.shape == (n_chnls,1)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples-1)    
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = one_way_test(data[:,0], labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
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
    assert np.isclose(p[0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
        
    
@pytest.mark.parametrize('method, result_p, result_obs, result_resmp',
                         [('permutation', (0.05,0.05,0.4), (3.10,24.91,0.18), (0.25,0.19,0.20))])                        
def test_two_way_test(two_way_data, method, result_p, result_obs, result_resmp):    
    """ Unit tests for two_way_test function for ANOVA-like 2-way stats """
    data, labels = two_way_data
    
    n = 10
    n_groups = 4
    n_terms = 3
    n_chnls = 4
    n_resamples = 20
    
    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity    
    p, stat_obs, stat_resmp = two_way_test(data, labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
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
    assert p2.shape == p.shape
    assert np.allclose(p, p2)
            
    # Test for consistent output with string-valued labels                                        
    groups = np.asarray(['cond1','cond2','cond3','cond4'])
    p2, stat_obs2, stat_resmp2 = two_way_test(data, groups[labels], axis=0, method=method, seed=1,
                                              n_resamples=n_resamples, return_stats=True)
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
    assert p.shape == (n_terms,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (n_terms,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_terms,n_chnls/2,n_chnls/2,n_resamples-1)    
    assert np.allclose(p[:,0,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[:,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[:,0,0,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)
                
    # Test for consistent output with transposed data dimensionality
    p, stat_obs, stat_resmp = two_way_test(data.T, labels, axis=-1, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert p.shape == (n_chnls,n_terms)
    assert stat_obs.shape == (n_chnls,n_terms)
    assert stat_resmp.shape == (n_chnls,n_terms,n_resamples-1)    
    assert np.allclose(p[0,:], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[0,:], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[0,:,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with vector-valued data
    p, stat_obs, stat_resmp = two_way_test(data[:,0], labels, axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert p.shape == (n_terms,)
    assert stat_obs.shape == (n_terms,)
    assert stat_resmp.shape == (n_terms,n_resamples-1)    
    assert np.allclose(p[:], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[:], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[:,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for expected output shape without interaction term call
    p, stat_obs, stat_resmp = two_way_test(data, labels[:,:-1], axis=0, method=method, seed=1,
                                           n_resamples=n_resamples, return_stats=True)
    assert p.shape == (n_terms-1,n_chnls)
    assert stat_obs.shape == (n_terms-1,n_chnls)
    assert stat_resmp.shape == (n_terms-1,n_chnls,n_resamples-1)
    
    # Test for consistent output with custom stat function (but setting it equal to defaults)
    stat_func = lambda data,labels: two_way_fstat(data, labels, axis=0)
    p, stat_obs, stat_resmp = two_way_test(data, labels, axis=0, method=method, stat=stat_func,
                                           seed=1, n_resamples=n_resamples, return_stats=True)
    assert np.allclose(p[:,0], result_p, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_obs[:,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.allclose(stat_resmp[:,0,:].mean(axis=-1), result_resmp, rtol=1e-2, atol=1e-2)
        
    
@pytest.mark.parametrize('method, result_ci, result_obs, result_resmp',
                         [('bootstrap', (8.53,12.18), 10.35, 10.36)])                        
def test_bootstrap_confints(one_sample_data, method, result_ci, result_obs, result_resmp):    
    """ Unit tests for bootstrap_confints function for 1-sample confidence intervals """    
    data = one_sample_data

    n = int(10)
    n_chnls = int(4)
    n_resamples = int(40)
    
    # Basic test of shape, value of output
    # Only test values for 1st simulated channel for simplicity    
    ci, stat_obs, stat_resmp = bootstrap_confints(data, axis=0, n_resamples=n_resamples,
                                                  seed=1, return_stats=True)
    assert ci.shape == (2,n_chnls)
    assert stat_obs.shape == (1,n_chnls)
    assert stat_resmp.shape == (n_resamples,n_chnls)
    print(ci.shape, stat_obs.shape, stat_resmp.shape)    
    print(ci[:,0], stat_obs[0,0], stat_resmp[:,0].mean())
    assert np.allclose(ci[:,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with return_stats=False call
    ci2 = bootstrap_confints(data, axis=0, n_resamples=n_resamples, seed=1, return_stats=False)
    assert ci2.shape == ci.shape
    assert np.allclose(ci, ci2)
        
    # Test for consistent output with different data array shape (3rd axis)
    ci, stat_obs, stat_resmp = bootstrap_confints(data.reshape((n,int(n_chnls/2),int(n_chnls/2))),
                                                  axis=0, n_resamples=n_resamples,
                                                  seed=1, return_stats=True)    
    assert ci.shape == (2,n_chnls/2,n_chnls/2)
    assert stat_obs.shape == (1,n_chnls/2,n_chnls/2)
    assert stat_resmp.shape == (n_resamples,n_chnls/2,n_chnls/2)    
    assert np.allclose(ci[:,0,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
                
    # Test for consistent output with transposed data dimensionality
    ci, stat_obs, stat_resmp = bootstrap_confints(data.T, axis=-1, n_resamples=n_resamples,
                                                  seed=1, return_stats=True)
    assert ci.shape == (n_chnls,2)
    assert stat_obs.shape == (n_chnls,1)
    assert stat_resmp.shape == (n_chnls,n_resamples)    
    assert np.allclose(ci[0,:], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[0,:].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with vector-valued data
    ci, stat_obs, stat_resmp = bootstrap_confints(data[:,0], axis=0, n_resamples=n_resamples,
                                                  seed=1, return_stats=True)    
    assert ci.shape == (2,)
    assert isinstance(stat_obs,float)
    assert stat_resmp.shape == (n_resamples,)    
    assert np.allclose(ci, result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs, result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with custom stat function (but setting it equal to defaults)
    stat_func = lambda data: np.mean(data, axis=0, keepdims=True)
    ci, stat_obs, stat_resmp = bootstrap_confints(data, axis=0, stat=stat_func, n_resamples=n_resamples,
                                                  seed=1, return_stats=True)
    print(ci.shape, stat_obs.shape, stat_resmp.shape)
    print(ci[:,0], stat_obs[0,0], stat_resmp[:,0].mean())
    assert np.allclose(ci[:,0], result_ci, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_obs[0,0], result_obs, rtol=1e-2, atol=1e-2)
    assert np.isclose(stat_resmp[:,0].mean(), result_resmp, rtol=1e-2, atol=1e-2)
    