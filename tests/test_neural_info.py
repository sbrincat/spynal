""" Unit tests for neural_info.py module """
import pytest
import numpy as np

from scipy.stats import norm

from ..neural_info import neural_info, neural_info_2groups

# TODO Test specific pev models, including multifactor ANOVA2,regress

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def gen_data():
    """ 
    Fixture simulates set of spike trains as spike timestamps to use for all unit tests 
    
    RETURNS
    data    (20,4) ndarray. Simulated data, simulating (10 trials*2 conditions x 4 channels)
    
    labels  (20,) ndarray of int8. Set of condition labels corresponding to each trial in data
    """    
    # Note: seed=1 makes data reproducibly match output of Matlab    
    np.random.seed(1)
    
    data = np.concatenate((norm.ppf(np.random.rand(10,4), loc=5.0, scale=5.0),
                           norm.ppf(np.random.rand(10,4), loc=10.0, scale=5.0)), axis=0)
    labels = np.hstack((np.zeros((10,),dtype='uint8'), np.ones((10,),dtype='uint8')))
    
    return data, labels


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('method, result',
                         [('pev', (6.26,-2.94,54.51,26.08)),
                          ('dprime', (-0.68,-0.29,-2.23,-1.27)),
                          ('auroc', (0.34,0.46,0.05,0.19)),
                          ('mutual_information', (0.56,0.40,0.90,0.76))])                        
def test_neural_info(gen_data, method, result):    
    """ Unit tests for neural_info function for computing neural information """    
    data, labels = gen_data
    
    # HACK For mutual information, must discretize data values
    if method == 'mutual_information': data = np.round(data)
    
    # Basic test of shape, value of output
    # Only information value for 1st simulated channel for simplicity    
    info = neural_info(labels, data, axis=0, method=method)
    assert info.shape == (1,4)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with 2-group form of neural computation function
    info2 = neural_info_2groups(data[labels==0,:], data[labels==1,:], axis=0, method=method)
    assert info2.shape == (1,4)
    assert np.allclose(info, info2)
    
    # Test for consistent output with different data array shape
    info = neural_info(labels, data.reshape((20,2,2)), axis=0, method=method)
    assert info.shape == (1,2,2)
    assert np.allclose(info.flatten().squeeze(), result, rtol=1e-2, atol=1e-2)
            
    # Test for consistent output with transposed data dimensionality
    info = neural_info(labels, data.T, axis=-1, method=method)
    assert info.shape == (4,1)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
    
    # Test for consistent output with vector-valued data
    info = neural_info(labels, data[:,0], axis=0, method=method)
    assert isinstance(info,float)
    assert np.isclose(info, result[0], rtol=1e-2, atol=1e-2)
                
            