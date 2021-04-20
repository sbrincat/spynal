""" Fixtures for generating synthetic data for unit tests """
import pytest
import numpy as np

from scipy.stats import norm

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
    
    labels  (40,4) ndarray of int8. Set of condition labels corresponding to each trial in data.
            Columns 0,1 correspond to main effects, column 2 corresponds to interaction    
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
