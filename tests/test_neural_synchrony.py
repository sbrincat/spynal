""" Unit tests for neural_synchrony.py module """
import pytest
import numpy as np

from scipy.stats import bernoulli

from ..neural_synchrony import simulate_oscillation, spectrum, power_spectrum, \
                               spectrogram, power_spectrogram, phase_spectrogram

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def oscillation():
    """ 
    Fixture simulates set of instances of oscillatory data for all unit tests 
    
    RETURNS
    data    (1001,4) ndarray. Simulated oscillatory data.
            (eg simulating 1001 timepoints x 4 trials or channels)    
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    frequency = 32 
    return simulate_oscillation(frequency, amplitude=5.0, phase=0, noise=1.0,
                                n_trials=4, time_range=1.0, smp_rate=1000, seed=1)


@pytest.fixture(scope='session')
def bursty_oscillation():
    """ 
    Fixture simulates set of instances of bursty oscillatory data for all unit tests 
    
    RETURNS
    data    (1001,4) ndarray. Simulated bursty oscillatory data.
            (eg simulating 1001 timepoints x 4 trials or channels)    
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    frequency = 32 
    return simulate_oscillation(frequency, amplitude=5.0, phase=0, noise=1.0, burst_rate=0.4,
                                time_range=1.0, n_trials=4, smp_rate=1000, seed=1)
    

@pytest.fixture(scope='session')
def spiking_oscillation(oscillation):
    """ 
    Fixture simulates set of instances of oscillatory spiking data for all unit tests 
    
    RETURNS
    data    (1001,4) ndarray of bool. Simulated oscillatory spiking data, 
            expressed as binary (0/1) spike trains.
            (eg simulating 1001 timepoints x 4 trials or channels)    
    """
    # TODO code up something actually proper (rate-modulated Poisson process?)
    data = oscillation
    
    # Convert continuous oscillation to probability (range 0-1)
    data = (data - data.min()) / data.ptp()
    data = data**2  # Sparsen high rates some
    
    # Use probabilities to generate Bernoulli random variable at each time point
    return bernoulli.ppf(0.5, data).astype(bool)
        

@pytest.fixture(scope='session')
def oscillatory_data(oscillation, bursty_oscillation, spiking_oscillation):
    """
    "Meta-fixture" that returns standard analog oscillations, bursty oscillations,
    and spiking oscillations in a single dict.
    
    RETURNS
    data_dict   {'data_type' : data} dict containing outputs from
                each of constituent fixtures
                
    SOURCE      https://stackoverflow.com/a/42400786
    """    
    return {'lfp': oscillation, 'burst': bursty_oscillation, 'spike':spiking_oscillation}



# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('data_type, method, result',
                         [('lfp', 'multitaper', 0.0137),
                          ('spike', 'multitaper', 194.0514)])    
def test_power_spectrum(oscillatory_data, data_type, method, result):
    """ Unit tests for power_spectrum function """
    # TODO Add wavelet, bandfilter methods
    data = oscillatory_data[data_type]
    smp_rate = 1000
    n_trials = 4
    n_freqs = {'wavelet': 26, 'multitaper':513,  'bandfilter': 3}
    spec_type = 'power'
        
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over all freqs in first trial for simplicity
    spec, freqs = spectrum(data, smp_rate, axis=0, method=method,
                                 data_type=data_type, spec_type=spec_type)
    print(spec.shape, np.round(spec[:,0].mean(),4))
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert spec.shape == (n_freqs[method], n_trials)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,0].mean(), result, rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs = spectrum(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                                 smp_rate, axis=0, method=method, data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert spec.shape == (n_freqs[method], n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,0,0].mean(), result, rtol=1e-4, atol=1e-4)
        
    # Test for consistent output with transposed data dimensionality
    spec, freqs = spectrum(data.T, smp_rate, axis=-1, method=method, data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert spec.shape == (n_trials, n_freqs[method])
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[0,:].mean(), result, rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with vector-valued data
    spec, freqs = spectrum(data[:,0], smp_rate, axis=-1, method=method,
                                 data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert spec.shape == (n_freqs[method],)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec.mean(), result, rtol=1e-4, atol=1e-4)

            
@pytest.mark.parametrize('data_type, method, result',
                         [('lfp', 'wavelet', 44.9462),
                          ('lfp', 'multitaper', 0.0137),
                          ('lfp', 'bandfilter', 2.8736),
                          ('spike', 'wavelet', 0.4774),
                          ('spike', 'multitaper', 193.7612),
                          ('spike', 'bandfilter', 0.0568)])
def test_power_spectrogram(oscillatory_data, data_type, method, result):
    """ Unit tests for power_spectrogram function """
    # Extract given data type from data dict
    data = oscillatory_data[data_type]    
    smp_rate = 1000
    n_trials = 4
    n_freqs = {'wavelet': 26, 'multitaper':257,  'bandfilter': 3}
    n_timepts = {'wavelet': 1000, 'multitaper':2,  'bandfilter': 1000}    
    spec_type = 'power'
    
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method,
                                       data_type=data_type, spec_type=spec_type)
    # print(spec.shape, np.round(spec[:,:,0].mean(),4))
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_freqs[method], n_timepts[method], n_trials)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,:,0].mean(), result, rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs, timepts = spectrogram(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                                       smp_rate, axis=0, method=method, data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_freqs[method], n_timepts[method], n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,:,0,0].mean(), result, rtol=1e-4, atol=1e-4)
 
    # Test for consistent output with transposed data dimensionality
    spec, freqs, timepts = spectrogram(data.T, smp_rate, axis=-1, method=method,
                                       data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_trials, n_freqs[method], n_timepts[method])
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[0,:,:].mean(), result, rtol=1e-4, atol=1e-4)
           
    # Test for consistent output with vector-valued data
    spec, freqs, timepts = spectrogram(data[:,0], smp_rate, axis=0, method=method,
                                       data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_freqs[method], n_timepts[method])
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)
    
    
@pytest.mark.parametrize('data_type, method, result',
                         [('lfp', 'wavelet', 0.0053),
                          ('lfp', 'multitaper', -0.1523),
                          ('lfp', 'bandfilter', 0.0016),
                          ('spike', 'wavelet', 0.0054),
                          ('spike', 'multitaper', -0.0547),
                          ('spike', 'bandfilter', 0.0649)])    
def test_phase_spectrogram(oscillatory_data, data_type, method, result):
    """ Unit tests for phase_spectrogram function """
    # Extract given data type from data dict
    data = oscillatory_data[data_type]       
    smp_rate = 1000
    n_trials = 4
    n_freqs = {'wavelet': 26, 'multitaper':257,  'bandfilter': 3}
    n_timepts = {'wavelet': 1000, 'multitaper':2,  'bandfilter': 1000}    
    spec_type = 'phase'
        
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over entire array -> scalar
    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method,
                                             data_type=data_type, spec_type=spec_type)
    print(data_type, method, spec.shape, np.round(spec[:,:,0].mean(),4))
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_freqs[method], n_timepts[method], n_trials)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,:,0].mean(), result, rtol=1e-4, atol=1e-4)
        
    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs, timepts = spectrogram(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                                             smp_rate, axis=0, method=method,
                                             data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_freqs[method], n_timepts[method], n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,:,0,0].mean(), result, rtol=1e-4, atol=1e-4)
 
    # Test for consistent output with transposed data dimensionality
    spec, freqs, timepts = spectrogram(data.T, smp_rate, axis=-1, method=method,
                                             data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_trials, n_freqs[method], n_timepts[method])
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[0,:,:].mean(), result, rtol=1e-4, atol=1e-4)
           
    # Test for consistent output with vector-valued data
    spec, freqs, timepts = spectrogram(data[:,0], smp_rate, axis=0, method=method,
                                             data_type=data_type, spec_type=spec_type)
    assert freqs.shape == (n_freqs[method],2) if method == 'bandfilter' else freqs.shape == (n_freqs[method],)
    assert timepts.shape == (n_timepts[method],)
    assert spec.shape == (n_freqs[method], n_timepts[method])
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)
    
        