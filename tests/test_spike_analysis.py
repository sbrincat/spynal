""" Unit tests for spike_analysis.py module """
import pytest
import numpy as np

from ..spike_analysis import simulate_spike_trains, setup_sliding_windows, \
                             times_to_bool, bool_to_times, bin_rate, density

# TODO  Tests for cut_trials, realign_spike_times, pool_electrode_units

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def spike_timestamp():
    """ 
    Fixture simulates set of spike trains as spike timestamps to use for all unit tests 
    
    RETURNS
    data    (10,2) ndarray of (n_spikes,) objects. Simulated spike timestamps
            (eg simulating 10 trials x 2 units)
    
    t       None. Second argout returned only to match output of spike_bool
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    data, _ = simulate_spike_trains(n_trials=20, n_conds=1, refractory=1e-3, seed=1)
    # Reshape data to simulate (n_trials,n_units) spiking data
    return data.reshape(10,2), None


@pytest.fixture(scope='session')
def spike_bool(spike_timestamp):
    """
    Fixture simulates set of spike trains as binary trains to use for all unit tests
    
    RETURNS
    data    (10,2,1001) ndarray of bool. Simulated binary spike trains
            (eg simulating 10 trials x 2 units x 1001 timepts).
    
    timepts (1001,) ndarray of float. Time sampling vector for data (in s).
    """
    # Note: Implicitly also tests times_to_bool function
    return times_to_bool(spike_timestamp[0], lims=[0,1])


@pytest.fixture(scope='session')
def spike_data(spike_timestamp, spike_bool):
    """
    "Meta-fixture" that returns both timestamp and boolean spiking data types
    in a dictionary.
    
    RETURNS
    data_dict   {'data_type' : (data,timepts)} dict containing outputs from
                each of constituent fixtures
                
    REFERENCE   https://stackoverflow.com/a/42400786
    """    
    return {'spike_timestamp': spike_timestamp, 'spike_bool': spike_bool}

             
# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('data_type, count, result',
                         [('spike_timestamp', True, 105),
                          ('spike_timestamp', False, 5.249999999999999),
                          ('spike_bool', True, 105),
                          ('spike_bool', False, 5.249999999999999)])                        
def test_bin_rate(spike_data, data_type, count, result):    
    """ Unit tests for bin_rate function for computing binned spike rates """    
    data, t = spike_data[data_type]
    
    # Basic test of shape, dtype, value of output. 
    # Test values summed over entire array -> scalar for spike counts
    # Test values averaged over entire array -> scalar for spike rates
    rates, bins = bin_rate(data, lims=[0,1], count=count, axis=-1, t=t)
    assert bins.shape == (20, 2)
    assert rates.shape == (10, 2, 20)
    assert np.issubdtype(rates.dtype,np.integer) if count else np.issubdtype(rates.dtype,np.float)
    assert np.isclose(rates.sum(), result) if count else np.isclose(rates.mean(), result)
    
    # Test for consistent output with different dimensionality
    shape = (5,2,2,*data.shape[2:])
    rates, bins = bin_rate(data.reshape(shape), lims=[0,1], count=count, axis=-1, t=t)
    assert bins.shape == (20, 2)
    assert rates.shape == (5, 2, 2, 20)
    assert np.isclose(rates.sum(), result) if count else np.isclose(rates.mean(), result)
    
    # Test for consistent output with different sliding window length
    rates, bins = bin_rate(data, lims=[0,1], width=20e-3, count=count, axis=-1, t=t)
    assert bins.shape == (50, 2)  
    assert rates.shape == (10, 2, 50)
    assert np.isclose(rates.sum(), result) if count else np.isclose(rates.mean(), result)
    
    # Test for consistent ouptut when bins are set explicitly
    bins = setup_sliding_windows(20e-3,[0,1])
    rates, bins = bin_rate(data, bins=bins, count=count, axis=-1, t=t)
    assert bins.shape == (50, 2)  
    assert rates.shape == (10, 2, 50)
    assert np.isclose(rates.sum(), result) if count else np.isclose(rates.mean(), result)
    
    # Test for consistent output for custom unequal-width bins
    # Note: Summed counts should be same, but mean rates are expected to be slightly different here
    bins = [[0,250e-3], [250e-3,750e-3], [750e-3,1000e-3]]
    rates, bins = bin_rate(data, bins=bins, count=count, axis=-1, t=t)
    assert rates.shape == (10, 2, 3)
    assert np.isclose(rates.sum(), result) if count else np.isclose(rates.mean(), 5.533333333333333)
    

@pytest.mark.parametrize('data_type, kernel, result',
                         [('spike_timestamp', 'gaussian', 4.91707260267567),
                          ('spike_timestamp', 'hanning', 4.929191169335072),
                          ('spike_bool', 'gaussian', 4.91707260267567),
                          ('spike_bool', 'hanning', 4.929191169335072)])                        
def test_density(spike_data, data_type, kernel, result):    
    """ Unit tests for bin_rate function for computing binned spike rates """    
    data, t = spike_data[data_type]
    # Set kernel width parameter so gaussian and hanning kernels are ~ identical
    width = 50e-3 if kernel == 'gaussian' else 2.53*50e-3

    # Basic test of shape, dtype, value of output
    # Test values averaged over entire array -> scalar for spike rates
    rates, tout = density(data, kernel=kernel, width=width, lims=[0,1], buffer=0, axis=-1, t=t)
    assert tout.shape == (1001,)
    assert rates.shape == (10, 2, 1001)
    assert np.isclose(rates.mean(), result)
    
    # Test for ~ consistent ouptut with 10x downsampling after spike density estimation
    rates, tout = density(data, kernel=kernel, width=width, lims=[0,1], buffer=0, 
                          downsmp=10, axis=-1, t=t)
    assert tout.shape == (101,)
    assert rates.shape == (10, 2, 101)
    assert np.isclose(rates.mean(), result, rtol=0.01, atol=0.01)
  
    
def test_bool_to_times(spike_timestamp, spike_bool):
    """ Unit tests for bool_to_times function to convert spike timestamps to binary trains """
    data_timestamp, _   = spike_timestamp
    data_bool, t        = spike_bool
    
    data_bool_to_timestamp = bool_to_times(data_bool, t, axis=-1)
    
    # Test that timestamp->bool->timestamp data retains same dtype, shape, values
    assert data_timestamp.dtype == data_bool_to_timestamp.dtype
    assert data_timestamp.shape == data_bool_to_timestamp.shape
    assert np.asarray([d1.shape == d2.shape for d1,d2 
                       in zip(data_timestamp.flatten(), data_bool_to_timestamp.flatten())]).all()
    assert np.asarray([np.allclose(np.round(d1,3), d2) for d1,d2 
                       in zip(data_timestamp.flatten(), data_bool_to_timestamp.flatten())]).all()
    

def test_times_to_bool(spike_timestamp, spike_bool):
    """ Unit tests for times_to_bool function to convert binary spike trains to timestamps """
    data_timestamp, _   = spike_timestamp
    data_bool, t        = spike_bool
    
    data_timestamp_to_bool, t2 = times_to_bool(data_timestamp, lims=(0,1))
    
    # Test that bool->timestamp->bool data retains same shape, dtype, values    
    assert (t == t2).all()
    assert data_bool.shape == data_timestamp_to_bool.shape
    assert data_bool.dtype == data_timestamp_to_bool.dtype
    assert (data_bool == data_timestamp_to_bool).all()    