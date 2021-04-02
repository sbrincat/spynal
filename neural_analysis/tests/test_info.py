""" Unit tests for info.py module """
import pytest
import numpy as np

from scipy.stats import norm

from ..info import neural_info, neural_info_2groups

# TODO Test specific pev models, including multifactor ANOVA2, regress and their stats output

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def gen_data():
    """ 
    Fixture simulates set of random data to use for all unit tests 
    
    RETURNS
    data    (20,4) ndarray. Simulated data, simulating (10 trials*2 conditions x 4 channels)
    
    labels  (20,) ndarray of int8. Set of condition labels corresponding to each trial in data
    """    
    # Note: seed=1 makes data reproducibly match output of Matlab    
    np.random.seed(1)
    
    n = 10
    n_chnls = 4    
    data = np.concatenate((norm.ppf(np.random.rand(n,n_chnls), loc=5.0, scale=5.0),
                           norm.ppf(np.random.rand(n,n_chnls), loc=10.0, scale=5.0)), axis=0)
    labels = np.hstack((np.zeros((n,),dtype='uint8'), np.ones((n,),dtype='uint8')))
    
    return data, labels


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('method, result',
                         [('pev', (6.26,-2.94,54.51,26.08)),
                          ('dprime', (-0.68,-0.29,-2.23,-1.27)),
                          ('auroc', (0.34,0.46,0.05,0.19)),
                          ('mutual_information', (0.26,0.06,0.72,0.44))])                        
def test_neural_info(gen_data, method, result):    
    """ Unit tests for neural_info function for computing neural information """    
    data, labels = gen_data
    
    n = int(10)
    n_chnls = int(4)
    
    # Basic test of shape, value of output
    info = neural_info(labels, data, axis=0, method=method)
    assert info.shape == (1,n_chnls)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with 2-group form of neural computation function
    info2 = neural_info_2groups(data[labels==0,:], data[labels==1,:], axis=0, method=method)
    assert info2.shape == (1,n_chnls)
    assert np.allclose(info, info2)
    
    # Test for consistent output with different data array shape (3rd axis)
    info = neural_info(labels, data.reshape((n*2,int(n_chnls/2),int(n_chnls/2))), axis=0, method=method)
    assert info.shape == (1,n_chnls/2,n_chnls/2)
    assert np.allclose(info.flatten().squeeze(), result, rtol=1e-2, atol=1e-2)
            
    # Test for consistent output with transposed data dimensionality
    info = neural_info(labels, data.T, axis=-1, method=method)
    assert info.shape == (n_chnls,1)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with vector-valued data
    # For mutual info, computing bins over all data != over 1 channel, 
    # so this ensures we have the same binning for both
    if method == 'mutual_information':
        bins = np.histogram_bin_edges(data, bins='fd')
        bins = np.stack((bins[:-1],bins[1:]),axis=1)
        extra_args = dict(bins=bins)
    else:
        extra_args = {}
    info = neural_info(labels, data[:,0], axis=0, method=method, **extra_args)
    assert isinstance(info,float)
    assert np.isclose(info, result[0], rtol=1e-2, atol=1e-2)
                 
    # Test for consistent output with string-valued labels                                        
    groups = np.asarray(['cond1','cond2'])
    info = neural_info(groups[labels], data, axis=0, method=method)
    assert info.shape == (1,n_chnls)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output using groups argument to subset data                                      
    info = neural_info(np.hstack((labels,labels+2)), np.concatenate((data,data),axis=0), 
                       axis=0, method=method, groups=[2,3])
    assert info.shape == (1,n_chnls)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
        