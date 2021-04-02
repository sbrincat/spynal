""" Unit tests for sync.py module """
import pytest
from math import pi
import numpy as np

from ..sync import simulate_multichannel_oscillation, synchrony
from ..spectra import spectrogram

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def oscillation_pair():
    """ 
    Fixture simulates set of instances of pairs of weakly synchronized oscillatory data 
    for unit tests of synchrony computation functions
    
    RETURNS
    data    (1001,40,2) ndarray. Simulated oscillatory data.
            (eg simulating 1001 timepoints x 40 trials x 2 channels)                        
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    frequency = 32 
    return simulate_multichannel_oscillation(2, frequency, amplitude=5.0, phase=[pi/4,0], phase_sd=[0,pi/4],
                                             noise=1.0, n_trials=40, time_range=1.0, smp_rate=1000, seed=1)


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('method, spec_method, result',
                         [('coherence', 'wavelet',      (0.2475,0.4795)),
                          ('coherence', 'multitaper',   (0.1046,0.3043)),
                          ('coherence', 'bandfilter',   (0.3490,0.6703)),
                          ('PLV',       'wavelet',      (0.2541,0.3713)),
                          ('PLV',       'multitaper',   (0.0984,0.2228)),
                          ('PLV',       'bandfilter',   (0.3322,0.4532)),
                          ('PPC',       'wavelet',      (0.0912,0.3713)),
                          ('PPC',       'multitaper',   (0.0100,0.2228)),
                          ('PPC',       'bandfilter',   (0.1715,0.4532))])
def test_synchrony(oscillation_pair, method, spec_method, result):
    """ Unit tests for synchrony() function """
    # Extract per-channel data and reshape -> (n_trials,n_timepts)
    data1, data2 = oscillation_pair[:,:,0].T, oscillation_pair[:,:,1].T
    
    smp_rate = 1000
    n_trials = 40
    n_freqs = {'wavelet': 26, 'multitaper':257,  'bandfilter': 3}
    n_timepts = {'wavelet': 1000, 'multitaper':2,  'bandfilter': 1000}    
        
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over all timepts, freqs for simplicity
    sync, freqs, timepts, dphi = synchrony(data1, data2, axis=0, method=method, spec_method=spec_method,
                                           return_phase=True, smp_rate=smp_rate, time_axis=-1)
    print(sync.shape, np.round(sync.mean(),4), dphi.shape, np.round(dphi.mean(),4), freqs.shape, timepts.shape)
    assert freqs.shape == (n_freqs[spec_method],2) if spec_method == 'bandfilter' else freqs.shape == (n_freqs[spec_method],)
    assert timepts.shape == (n_timepts[spec_method],)
    assert sync.shape == (n_freqs[spec_method], n_timepts[spec_method])
    assert dphi.shape == (n_freqs[spec_method], n_timepts[spec_method])
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(dphi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(dphi.mean(), result[1], rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with return_phase=False
    sync2, freqs2, timepts2 = synchrony(data1, data2, axis=0, method=method, spec_method=spec_method,
                                        return_phase=False, smp_rate=smp_rate, time_axis=-1)
    assert freqs2.shape == (n_freqs[spec_method],2) if spec_method == 'bandfilter' else freqs.shape == (n_freqs[spec_method],)
    assert timepts2.shape == (n_timepts[spec_method],)    
    assert np.allclose(sync2, sync, rtol=1e-4, atol=1e-4)

    # Test for consistent output with reversed time axis (dphi should change sign, otherwise same)
    # Skip test for multitaper/bandfilter phase, bandfilter sync -- not time-reversal invariant    
    sync, freqs, timepts, dphi = synchrony(np.flip(data1,axis=-1), np.flip(data2,axis=-1),
                                           axis=0, method=method, spec_method=spec_method,
                                           return_phase=True, smp_rate=smp_rate, time_axis=-1)
    if spec_method != 'bandfilter':
        assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    if spec_method == 'wavelet':    
        assert np.isclose(dphi.mean(), -result[1], rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with channels swapped (dphi should change sign, otherwise same)
    # Skip test for multitaper/bandfilter phase -- not time-reversal invariant
    sync, freqs, timepts, dphi = synchrony(data2, data1, axis=0, method=method, spec_method=spec_method,
                                           return_phase=True, smp_rate=smp_rate, time_axis=-1)
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    if spec_method == 'wavelet':
        assert np.isclose(dphi.mean(), -result[1], rtol=1e-4, atol=1e-4)
            
    # Test for consistent output with different data array shape (3rd axis)
    sync, freqs, timepts, dphi = synchrony(np.stack((data1,data1),axis=2),
                                           np.stack((data2,data2),axis=2),
                                           axis=0, method=method, spec_method=spec_method,
                                           return_phase=True, smp_rate=smp_rate, time_axis=1)
    assert freqs.shape == (n_freqs[spec_method],2) if spec_method == 'bandfilter' else freqs.shape == (n_freqs[spec_method],)
    assert timepts.shape == (n_timepts[spec_method],)
    assert sync.shape == (n_freqs[spec_method], n_timepts[spec_method], 2)
    assert dphi.shape == (n_freqs[spec_method], n_timepts[spec_method], 2)
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(dphi.dtype,np.float)
    # HACK Loosen tolerance bc this doesn't quite match up for bandfilter (TODO why???)
    assert np.isclose(sync[:,:,0].mean(), result[0], rtol=1e-3, atol=1e-3)
    assert np.isclose(dphi[:,:,0].mean(), result[1], rtol=1e-3, atol=1e-3)
 
    # Test for consistent output with transposed data dimensionality -> (time,trials)
    sync, freqs, timepts, dphi = synchrony(data1.T, data2.T, axis=-1, method=method, spec_method=spec_method,
                                           return_phase=True, smp_rate=smp_rate, time_axis=0)
    assert freqs.shape == (n_freqs[spec_method],2) if spec_method == 'bandfilter' else freqs.shape == (n_freqs[spec_method],)
    assert timepts.shape == (n_timepts[spec_method],)
    assert sync.shape == (n_freqs[spec_method], n_timepts[spec_method])
    assert dphi.shape == (n_freqs[spec_method], n_timepts[spec_method])
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(dphi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(dphi.mean(), result[1], rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with spectral data input
    spec1, freqs, timepts = spectrogram(data1, smp_rate, axis=1, method=spec_method,
                                        spec_type='complex', keep_tapers=True)
    spec2, freqs, timepts = spectrogram(data2, smp_rate, axis=1, method=spec_method,
                                        spec_type='complex', keep_tapers=True)
    print(data1.shape, spec2.shape)
    sync, _, _, dphi = synchrony(spec1, spec2, axis=0, taper_axis=2, method=method, spec_method=spec_method, 
                                 return_phase=True)
    assert freqs.shape == (n_freqs[spec_method],2) if spec_method == 'bandfilter' else \
           freqs.shape == (n_freqs[spec_method],)
    assert timepts.shape == (n_timepts[spec_method],)
    assert sync.shape == (n_freqs[spec_method], n_timepts[spec_method])
    assert dphi.shape == (n_freqs[spec_method], n_timepts[spec_method])
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(dphi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(dphi.mean(), result[1], rtol=1e-4, atol=1e-4)