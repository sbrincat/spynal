""" Unit tests for neural_synchrony.py module """
import pytest
import numpy as np

from ..neural_synchrony import simulate_oscillation, power_spectrum, power_spectrogram, \
                               phase_spectrogram


# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def oscillation():
    """ 
    Fixture simulates set of instances of oscillatory data for all unit tests 
    
    RETURNS
    data    (1001,2,10) ndarray. Simulated oscillatory data.
            (eg simulating 1001 timepoints x 2 channels x 10 trials)    
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    frequency = 32 
    data = simulate_oscillation(frequency, amplitude=5.0, phase=0, noise=1.0,
                                n_trials=20, time_range=1.0, smp_rate=1000, seed=1)
    
    # Reshape data to simulate (n_timepts,n_channels,n_trials) spiking data
    return data.reshape((-1,2,10))


@pytest.fixture(scope='session')
def bursty_oscillation():
    """ 
    Fixture simulates set of instances of bursty oscillatory data for all unit tests 
    
    RETURNS
    data    (1001,2,10) ndarray. Simulated bursty oscillatory data.
            (eg simulating 1001 timepoints x 2 channels x 10 trials)    
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    frequency = 32 
    data = simulate_oscillation(frequency, amplitude=5.0, phase=0, noise=1.0, burst_rate=0.4,
                                time_range=1.0, n_trials=20, smp_rate=1000, seed=1)
    
    # Reshape data to simulate (n_timepts,n_channels,n_trials) spiking data
    return data.reshape((-1,2,10))


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('signal, method, result',
                         [('lfp', 'multitaper', 0.013427687302890581)])    
def test_power_spectrum(oscillation, signal, method, result):
    """ Unit tests for power_spectrum function """
    # TODO Add wavelet, bandfilter methods and spike signals
    smp_rate = 1000
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over entire array -> scalar
    spec, freqs = power_spectrum(oscillation, smp_rate, axis=0,
                                 method=method, signal=signal)
    print(spec.shape, spec.mean())
    shape = {'wavelet'      : (26, 2, 10), 
             'multitaper'   : (513, 2, 10), 
             'bandfilter'   : (3, 2, 10)}
    assert freqs.shape == (shape[method][0],)
    assert spec.shape == shape[method]
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec.mean(), result)
    
    
@pytest.mark.parametrize('signal, method, result',
                         [('lfp', 'wavelet', 44.13694364696586),
                          ('lfp', 'multitaper', 0.013397496009470008),
                          ('lfp', 'bandfilter', 2.8076167074041734)])    
def test_power_spectrogram(oscillation, signal, method, result):
    """ Unit tests for power_spectrogram function """
    # TODO Add spike signals
    smp_rate = 1000
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over entire array -> scalar
    spec, freqs, timepts = power_spectrogram(oscillation, smp_rate, axis=0,
                                             method=method, signal=signal)
    print(spec.shape, spec.mean())
    shape = {'wavelet'      : (26, 1000, 2, 10), 
             'multitaper'   : (257, 2, 2, 10), 
             'bandfilter'   : (3, 1000, 2, 10)}
    assert freqs.shape == (shape[method][0],)
    assert timepts.shape == (shape[method][1],)
    assert spec.shape == shape[method]
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec.mean(), result)
    
    
@pytest.mark.parametrize('signal, method, result',
                         [('lfp', 'wavelet', 0.00031825649435524494),
                          ('lfp', 'multitaper', 0.014038059049161174),
                          ('lfp', 'bandfilter', 0.09190707938093082)])    
def test_phase_spectrogram(oscillation, signal, method, result):
    """ Unit tests for phase_spectrogram function """
    smp_rate = 1000
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over entire array -> scalar
    spec, freqs, timepts = phase_spectrogram(oscillation, smp_rate, axis=0,
                                             method=method, signal=signal)
    print(spec.shape, spec.mean())
    shape = {'wavelet'      : (26, 1000, 2, 10), 
             'multitaper'   : (257, 2, 2, 10), 
             'bandfilter'   : (3, 1000, 2, 10)}
    assert freqs.shape == (shape[method][0],)
    assert timepts.shape == (shape[method][1],)
    assert spec.shape == shape[method]
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec.mean(), result)
        
    