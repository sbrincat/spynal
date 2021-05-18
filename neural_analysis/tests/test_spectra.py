""" Unit tests for spectra.py module """
import pytest
import numpy as np

from scipy.stats import bernoulli

from ..spectra import simulate_oscillation, spectrum, power_spectrum, \
                      spectrogram, power_spectrogram, phase_spectrogram, \
                      cut_trials, realign_data

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def oscillation():
    """ 
    Fixture simulates set of instances of oscillatory data for all unit tests 
    
    RETURNS
    data    (1000,4) ndarray. Simulated oscillatory data.
            (eg simulating 1000 timepoints x 4 trials or channels)    
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
    data    (1000,4) ndarray. Simulated bursty oscillatory data.
            (eg simulating 1000 timepoints x 4 trials or channels)    
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
    data    (1000,4) ndarray of bool. Simulated oscillatory spiking data, 
            expressed as binary (0/1) spike trains.
            (eg simulating 1000 timepoints x 4 trials or channels)    
    """
    # todo code up something actually proper (rate-modulated Poisson process?)
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
# Unit tests for spectral analysis functions
# =============================================================================
@pytest.mark.parametrize('data_type, spec_type, method, result',
                         [('lfp',   'power', 'multitaper',  0.0137),
                          ('lfp',   'power', 'wavelet',     44.9462),
                          ('lfp',   'power', 'bandfilter',  2.3069),
                          ('spike', 'power', 'multitaper',  194.0514),
                          ('spike', 'power', 'wavelet',     0.4774),
                          ('spike', 'power', 'bandfilter',  0.0483)])
def test_spectrum(oscillatory_data, data_type, spec_type, method, result):
    """ Unit tests for spectrum function """
    data = oscillatory_data[data_type]
    smp_rate = 1000
    n_trials = 4
    method_to_n_freqs   = {'wavelet': 26, 'multitaper':513,  'bandfilter': 3}
    n_freqs = method_to_n_freqs[method]
    freqs_shape = (n_freqs,2) if (method == 'bandfilter') or (spec_type == 'burst') else (n_freqs,)
        
    # Time reversal -> inverted sign phase, complex conj of complex, preserves power
    if spec_type == 'phase':        reversed_result = -result
    elif spec_type == 'complex':    reversed_result = np.conj(result)
    else:                           reversed_result = result
            
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over all freqs in first trial for simplicity
    spec, freqs = spectrum(data, smp_rate, axis=0, method=method, data_type=data_type, spec_type=spec_type)
    # print(spec.sape, np.round(spec[:,0].mean(),4))
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs, n_trials)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,0].mean(), result, rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs = spectrum(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                           smp_rate, axis=0, method=method, data_type=data_type, spec_type=spec_type)
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs, n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[:,0,0].mean(), result, rtol=1e-4, atol=1e-4)
        
    # Test for consistent output with transposed data dimensionality
    spec, freqs = spectrum(data.T, smp_rate, axis=-1, method=method, data_type=data_type, spec_type=spec_type)
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_trials, n_freqs)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec[0,:].mean(), result, rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with vector-valued data
    spec, freqs = spectrum(data[:,0], smp_rate, axis=-1, method=method, data_type=data_type, spec_type=spec_type)
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs,)
    assert np.issubdtype(spec.dtype,np.float)
    assert np.isclose(spec.mean(), result, rtol=1e-4, atol=1e-4)
           
    # Test for expected output with time-reversed data
    # Skip test for bandfilter method -- different initial conditions do change results slightly at start
    # Skip test for multitaper phase/complex -- not time-reversal invariant
    if (method == 'wavelet') or ((method == 'multitaper') and (spec_type == 'power')):
        spec, freqs = spectrum(np.flip(data,axis=0), smp_rate, axis=0, method=method,
                               data_type=data_type, spec_type=spec_type)
        assert spec.shape == (n_freqs, n_trials)
        assert np.isclose(spec[:,0].mean(), reversed_result, rtol=1e-4, atol=1e-4)
        
                                      
# TODO Take a closer look at 'complex' outputs -- why coming out = 0?            
@pytest.mark.parametrize('data_type, spec_type, method, result',
                         [('lfp',   'power',    'wavelet',     44.9462),
                          ('lfp',   'power',    'multitaper',  0.0137),
                          ('lfp',   'power',    'bandfilter',  2.3069),
                          ('burst', 'burst',    'wavelet',     0.0),
                          ('burst', 'burst',    'bandfilter',  0.0),
                          ('spike', 'power',    'wavelet',     0.4774),
                          ('spike', 'power',    'multitaper',  193.7612),
                          ('spike', 'power',    'bandfilter',  0.0483),
                          ('lfp',   'phase',    'wavelet',     0.0053),
                          ('lfp',   'phase',    'multitaper', -0.1523),
                          ('lfp',   'phase',    'bandfilter',  0.1473),
                          ('spike', 'phase',    'wavelet',     0.0054),
                          ('spike', 'phase',    'multitaper', -0.0547),
                          ('spike', 'phase',    'bandfilter',  0.1190),
                          ('lfp',   'complex',  'wavelet',     0j),
                          ('lfp',   'complex',  'multitaper',  0.0019-0.0011j),
                          ('lfp',   'complex',  'bandfilter',  -0.0109+0j)])
def test_spectrogram(oscillatory_data, data_type, spec_type, method, result):
    """ Unit tests for spectrogram() function """
    # Extract given data type from data dict
    data = oscillatory_data[data_type]
    data_type_ = 'lfp' if data_type == 'burst' else data_type
    smp_rate = 1000
    n_trials = 4
    
    method_to_n_freqs   = {'wavelet': 26, 'multitaper':257, 'bandfilter': 3, 'burst':4}
    method_to_n_timepts = {'wavelet': 1000, 'multitaper':2, 'bandfilter': 1000}
    n_freqs = method_to_n_freqs['burst'] if spec_type == 'burst' else method_to_n_freqs[method]
    freqs_shape = (n_freqs,2) if (method == 'bandfilter') or (spec_type == 'burst') else (n_freqs,)
    n_timepts = method_to_n_timepts[method]
    if spec_type == 'complex':  dtype = np.complex
    elif spec_type == 'burst':  dtype = np.bool
    else:                       dtype = np.float
    
    # Time reversal -> inverted sign phase, complex conj of complex, preserves power
    if spec_type == 'phase':        reversed_result = -result
    elif spec_type == 'complex':    reversed_result = np.conj(result)
    else:                           reversed_result = result
    
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method,
                                       data_type=data_type_, spec_type=spec_type)
    # print(spec.shape, np.round(spec[:,:,0].mean(),4), spec[:,:,0].mean())
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts, n_trials)
    assert np.issubdtype(spec.dtype,dtype)
    assert np.isclose(spec[:,:,0].mean(), result, rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs, timepts = spectrogram(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                                       smp_rate, axis=0, method=method, data_type=data_type_,
                                       spec_type=spec_type)
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts, n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,dtype)
    assert np.isclose(spec[:,:,0,0].mean(), result, rtol=1e-4, atol=1e-4)
 
    # Test for consistent output with transposed data dimensionality
    extra_args = {'trial_axis':0} if spec_type == 'burst' else {}
    spec, freqs, timepts = spectrogram(data.T, smp_rate, axis=-1, method=method,
                                       data_type=data_type_, spec_type=spec_type, **extra_args)
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_trials, n_freqs, n_timepts)
    assert np.issubdtype(spec.dtype,dtype)
    assert np.isclose(spec[0,:,:].mean(), result, rtol=1e-4, atol=1e-4)
           
    # Test for consistent output with vector-valued (single-trial/channel) data
    # Note: Skip for burst_analysis bc needs multiple trials to compute z-scores
    if spec_type != 'burst':
        spec, freqs, timepts = spectrogram(data[:,0], smp_rate, axis=0, method=method,
                                        data_type=data_type_, spec_type=spec_type)
        assert freqs.shape == freqs_shape
        assert timepts.shape == (n_timepts,)
        assert spec.shape == (n_freqs, n_timepts)
        assert np.issubdtype(spec.dtype,dtype)
        assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)
    
    # Test for expected output with time-reversed data
    # Skip test for bandfilter method -- different initial conditions do change results slightly at start
    # Skip test for multitaper phase/complex -- not time-reversal invariant
    if (method == 'wavelet') or ((method == 'multitaper') and (spec_type == 'power')):
        spec, freqs, timepts = spectrogram(np.flip(data,axis=0), smp_rate, axis=0, method=method,
                                        data_type=data_type_, spec_type=spec_type)
        # print(spec.shape, np.round(spec[:,:,0].mean(),4))    
        assert spec.shape == (n_freqs, n_timepts, n_trials)
        assert np.isclose(spec[:,:,0].mean(), reversed_result, rtol=1e-4, atol=1e-4)
        
  
# =============================================================================
# Unit tests for preprocessing/utility functions
# =============================================================================        
def test_cut_trials(oscillation):    
    """ Unit tests for cut_trials function """
    n_timepts, n_trials = oscillation.shape
    
    # Unwrap trial-cut data into one long vector to simulate uncut data
    uncut_data  = oscillation.flatten(order='F') 
    data        = oscillation
    
    # Check if unwrapped->recut data is same as original uncut data
    trial_lims  = np.asarray([0,0.999])[np.newaxis,:] + np.arange(n_trials)[:,np.newaxis]     
    cut_data    = cut_trials(uncut_data, trial_lims, smp_rate=1000, axis=0)
    assert cut_data.shape == data.shape
    assert (cut_data == data).all()

            
def test_realign_data(oscillation):    
    """ Unit tests for realign_data function """
    data = oscillation
    n_timepts, n_trials = data.shape
    timepts = np.arange(0,1,1e-3)

    # Realign to 2 distinct times, then concatenate together and test if same  
    realigned1 = realign_data(data, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                timepts=timepts, time_axis=0, trial_axis=-1)
    realigned2 = realign_data(data, 0.5*np.ones((n_trials,)), time_range=(0,0.499),
                                timepts=timepts, time_axis=0, trial_axis=-1)
    realigned = np.concatenate((realigned1,realigned2), axis=0)
    assert realigned.shape == data.shape
    assert (realigned == data).all()
    
    # Test for consistent output with transposed data dimensionality
    realigned1 = realign_data(data.T, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                timepts=timepts, time_axis=-1, trial_axis=0)
    realigned2 = realign_data(data.T, 0.5*np.ones((n_trials,)), time_range=(0,0.499),
                                timepts=timepts, time_axis=-1, trial_axis=0)
    realigned = np.concatenate((realigned1,realigned2), axis=-1)
    assert realigned.shape == data.T.shape
    assert (realigned == data.T).all()        
        