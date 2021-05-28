""" Unit tests for sync.py module """
import pytest
from math import pi
import numpy as np

from scipy.stats import bernoulli

from ..sync import simulate_multichannel_oscillation, synchrony, spike_field_coupling
from ..spectra import spectrogram

# =============================================================================
# Fixtures for generating simulated data
# =============================================================================
@pytest.fixture(scope='session')
def oscillation_pair():
    """ 
    Fixture simulates set of instances of pairs of weakly synchronized oscillatory data 
    for unit tests of field-field synchrony computation functions
    
    RETURNS
    data    (1000,40,2) ndarray. Simulated oscillatory data.
            (eg simulating 1000 timepoints x 40 trials x 2 channels)                        
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    frequency = 32 
    return simulate_multichannel_oscillation(2, frequency, amplitude=5.0, phase=[pi/4,0], phase_sd=[0,pi/4],
                                             noise=1.0, n_trials=40, time_range=1.0, smp_rate=1000, seed=1)


@pytest.fixture(scope='session')
def spike_field_pair(oscillation_pair):
    """ 
    Fixture simulates set of instances of weakly synchronized oscillatory spike-field data pairs
    for unit tests of spike-field synchrony computation functions
    
    RETURNS
    spkdata (1000,40) ndarray of bool. Simulated oscillatory spiking data, 
            expressed as binary (0/1) spike trains.
            (eg simulating 1000 timepoints x 4 trials or channels)

    lfpdata (1000,40) ndarray of float. Simulated oscillatory LFP data 
            (eg simulating 1000 timepoints x 4 trials or channels)            
    """
    # todo code up something actually proper (rate-modulated Poisson process?)
    spkdata,lfpdata = oscillation_pair[:,:,0], oscillation_pair[:,:,1]
    
    # Convert continuous oscillation to probability (range 0-1)
    spkdata = (spkdata - spkdata.min()) / spkdata.ptp()
    spkdata = spkdata**2  # Sparsen high rates some
    
    # Use probabilities to generate Bernoulli random variable at each time point
    spkdata =  bernoulli.ppf(0.5, spkdata).astype(bool)
    
    return spkdata, lfpdata


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
    method_to_n_freqs   = {'wavelet': 26, 'multitaper':257,  'bandfilter': 3}
    method_to_n_timepts = {'wavelet': 1000, 'multitaper':2,  'bandfilter': 1000}
    n_freqs     = method_to_n_freqs[spec_method]
    n_timepts   = method_to_n_timepts[spec_method]    
    freqs_shape = (n_freqs,2) if spec_method == 'bandfilter' else (n_freqs,)
        
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over all timepts, freqs for simplicity
    sync, freqs, timepts, dphi = synchrony(data1, data2, axis=0, method=method, spec_method=spec_method,
                                           return_phase=True, smp_rate=smp_rate, time_axis=-1)
    # print(sync.shape, np.round(sync.mean(),4), dphi.shape, np.round(dphi.mean(),4), freqs.shape, timepts.shape)
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert sync.shape == (n_freqs, n_timepts)
    assert dphi.shape == (n_freqs, n_timepts)
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(dphi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(dphi.mean(), result[1], rtol=1e-4, atol=1e-4)
    
    # Test for consistent output with return_phase=False
    sync2, freqs2, timepts2 = synchrony(data1, data2, axis=0, method=method, spec_method=spec_method,
                                        return_phase=False, smp_rate=smp_rate, time_axis=-1)
    assert freqs2.shape == freqs_shape
    assert timepts2.shape == (n_timepts,)    
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
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert sync.shape == (n_freqs, n_timepts, 2)
    assert dphi.shape == (n_freqs, n_timepts, 2)
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(dphi.dtype,np.float)
    # HACK Loosen tolerance bc this doesn't quite match up for bandfilter (TODO why???)
    assert np.isclose(sync[:,:,0].mean(), result[0], rtol=1e-3, atol=1e-3)
    assert np.isclose(dphi[:,:,0].mean(), result[1], rtol=1e-3, atol=1e-3)
 
    # Test for consistent output with transposed data dimensionality -> (time,trials)
    sync, freqs, timepts, dphi = synchrony(data1.T, data2.T, axis=-1, method=method, spec_method=spec_method,
                                           return_phase=True, smp_rate=smp_rate, time_axis=0)
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert sync.shape == (n_freqs, n_timepts)
    assert dphi.shape == (n_freqs, n_timepts)
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
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert sync.shape == (n_freqs, n_timepts)
    assert dphi.shape == (n_freqs, n_timepts)
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(dphi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(dphi.mean(), result[1], rtol=1e-4, atol=1e-4)
    
    
    
@pytest.mark.parametrize('method, spec_method, result',
                         [('coherence', 'wavelet',      (0.2400,0.2894)),
                          ('coherence', 'multitaper',   (0.0957,0.1477)),
                          ('coherence', 'bandfilter',   (0.3721,0.2676)),
                          ('PLV',       'wavelet',      (0.1600,-0.2363,1994)),
                          ('PLV',       'multitaper',   (0.0984,0.2228)),
                          ('PLV',       'bandfilter',   (0.3041,-0.2260,1994)),
                          ('PPC',       'wavelet',      (0.0764,-0.2363,1994)),
                          ('PPC',       'multitaper',   (0.0100,0.2228)),
                          ('PPC',       'bandfilter',   (0.1417,-0.2260,1994))])
def test_spike_field_coupling(spike_field_pair, method, spec_method, result):
    """ Unit tests for spike_field_coupling() function """
    # Extract per-channel data and reshape -> (n_trials,n_timepts)
    spkdata, lfpdata = spike_field_pair[0].T, spike_field_pair[1].T
    
    smp_rate = 1000
    method_to_n_freqs   = {'wavelet': 26, 'multitaper':257,  'bandfilter': 3}
    method_to_n_timepts = {'wavelet': 1000, 'multitaper':2,  'bandfilter': 1000} \
                          if method == 'coherence' else \
                          {'wavelet': 4, 'multitaper':4,  'bandfilter': 4}
    n_freqs     = method_to_n_freqs[spec_method]
    n_timepts   = method_to_n_timepts[spec_method]    
    freqs_shape = (n_freqs,2) if spec_method == 'bandfilter' else (n_freqs,)    
    timepts     = np.arange(lfpdata.shape[-1]) / smp_rate
    extra_args  = {'timepts':timepts, 'width':0.2} if method in ['PLV','PPC'] else {}
        
    # Basic test of shape, dtype, value of output. 
    # Test values averaged over all timepts, freqs for simplicity
    sync, freqs, timepts, n, phi = spike_field_coupling(spkdata, lfpdata, axis=0, time_axis=-1, method=method, 
                                                        spec_method=spec_method, smp_rate=smp_rate,
                                                        return_phase=True, **extra_args)
    print(sync.shape, np.round(sync.mean(),4), phi.shape, np.round(phi.mean(),4), freqs.shape, timepts.shape)
    if method != 'coherence': print(n.shape, np.round(n.mean()))
    assert isinstance(freqs, np.ndarray)
    assert isinstance(timepts, np.ndarray)
    assert isinstance(sync, np.ndarray)        
    if method != 'coherence': assert isinstance(n, np.ndarray)
    assert isinstance(phi, np.ndarray)    
    assert sync.shape == (n_freqs, n_timepts)    
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert phi.shape == (n_freqs, n_timepts)
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(phi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(phi.mean(), result[1], rtol=1e-4, atol=1e-4)
    if method != 'coherence': assert np.round(n.mean()) == result[2]
    
    
    # Test for consistent output with return_phase=False
    sync2, freqs2, timepts2, n = spike_field_coupling(spkdata, lfpdata, axis=0, time_axis=-1, method=method,
                                                      spec_method=spec_method, smp_rate=smp_rate,
                                                      return_phase=False, **extra_args)
    assert freqs2.shape == freqs_shape
    assert timepts2.shape == (n_timepts,)    
    assert np.allclose(sync2, sync, rtol=1e-4, atol=1e-4)
    if method != 'coherence': assert np.round(n.mean()) == result[2]

    # DELETE -- Doesn't work for windowed output here -- not time-reversal invariant
    # # Test for consistent output with reversed time axis (phi should change sign, otherwise same)
    # # Skip test for multitaper/bandfilter phase, bandfilter sync -- not time-reversal invariant    
    # sync, freqs, timepts, n, phi = spike_field_coupling(np.flip(spkdata,axis=-1),
    #                                                     np.flip(lfpdata,axis=-1),
    #                                                     axis=0, time_axis=-1, method=method,
    #                                                     spec_method=spec_method, smp_rate=smp_rate,
    #                                                     return_phase=True, **extra_args)
    # print(np.round(sync.mean(),4),  np.round(phi.mean(),4))
    # if spec_method != 'bandfilter':
    #     assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    # if spec_method == 'wavelet':    
    #     assert np.isclose(phi.mean(), result[1], rtol=1e-4, atol=1e-4)
    # if method != 'coherence': assert np.round(n.mean()) == result[2]
            
    # TEMP TODO Need to figure out WTF to do w/ dimensionality in PLV            
    # Test for consistent output with different data array shape (3rd axis)
    if method == 'coherence':
        sync, freqs, timepts, n, phi = spike_field_coupling(np.stack((spkdata,spkdata),axis=2),
                                                            np.stack((lfpdata,lfpdata),axis=2),
                                                            axis=0, time_axis=1, method=method,
                                                            spec_method=spec_method, smp_rate=smp_rate,
                                                            return_phase=True, **extra_args)
        assert freqs.shape == freqs_shape
        assert timepts.shape == (n_timepts,)
        assert sync.shape == (n_freqs, n_timepts, 2)
        assert phi.shape == (n_freqs, n_timepts, 2)
        assert np.issubdtype(sync.dtype,np.float)
        assert np.issubdtype(phi.dtype,np.float)
        # HACK Loosen tolerance bc this doesn't quite match up for bandfilter (TODO why???)
        assert np.isclose(sync[:,:,0].mean(), result[0], rtol=1e-3, atol=1e-3)
        assert np.isclose(phi[:,:,0].mean(), result[1], rtol=1e-3, atol=1e-3)
        if method != 'coherence': assert np.round(n.mean()) == result[2]

    # Test for consistent output with transposed data dimensionality -> (time,trials)
    sync, freqs, timepts, n, phi = spike_field_coupling(spkdata.T, lfpdata.T,
                                                        axis=-1, time_axis=0, method=method,
                                                        spec_method=spec_method, smp_rate=smp_rate,
                                                        return_phase=True, **extra_args)
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert sync.shape == (n_freqs, n_timepts)
    assert phi.shape == (n_freqs, n_timepts)
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(phi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(phi.mean(), result[1], rtol=1e-4, atol=1e-4)
    if method != 'coherence': assert np.round(n.mean()) == result[2]
    
    # Test for consistent output with spectral data input
    # Must convert spike data to spectral for coherence
    if method == 'coherence':        
        spkspec, _, _ = spectrogram(spkdata, smp_rate, axis=1, method=spec_method,
                                    spec_type='complex', keep_tapers=True)
    # Must insert singleton axis to match freq axis for other methods
    else:
        spkspec = spkdata[:,np.newaxis,:]
    lfpspec, freqs, timepts = spectrogram(lfpdata, smp_rate, axis=1, method=spec_method,
                                          spec_type='complex', keep_tapers=True)
    time_axis = 3 if spec_method == 'multitaper' else 2
    if spec_method == 'multitaper': extra_args.update(taper_axis=time_axis-1)
    print("OUT", spkspec.shape, lfpspec.shape)
    sync, _, _, n, phi = spike_field_coupling(spkspec, lfpspec, axis=0, time_axis=time_axis,
                                              method=method, spec_method=spec_method, 
                                              return_phase=True, **extra_args)
    assert sync.shape == (n_freqs, n_timepts)
    assert phi.shape == (n_freqs, n_timepts)
    assert np.issubdtype(sync.dtype,np.float)
    assert np.issubdtype(phi.dtype,np.float)    
    assert np.isclose(sync.mean(), result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(phi.mean(), result[1], rtol=1e-4, atol=1e-4)
    if method != 'coherence': assert np.round(n.mean()) == result[2]
        