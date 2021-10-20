""" Unit tests for spectra.py module """
import pytest
import numpy as np
import xarray as xr

from collections import OrderedDict
from scipy.stats import bernoulli

from neural_analysis.spectra import simulate_oscillation, spectrum, power_spectrum, \
                                    spectrogram, power_spectrogram, phase_spectrogram, itpc, \
                                    cut_trials, realign_data, pool_freq_bands, pool_time_epochs

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
    data_orig = data.copy()
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
    spec, freqs = spectrum(data, smp_rate, axis=0, method=method, data_type=data_type,
                           spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs, n_trials)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec[:,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs = spectrum(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                           smp_rate, axis=0, method=method, data_type=data_type,
                           spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs, n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec[:,0,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    spec, freqs = spectrum(data.T, smp_rate, axis=-1, method=method, data_type=data_type,
                           spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_trials, n_freqs)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec[0,:].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    spec, freqs = spectrum(data[:,0], smp_rate, axis=-1, method=method, data_type=data_type,
                           spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs,)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec.mean(), result, rtol=1e-4, atol=1e-4)

    # Test for expected output with time-reversed data
    # Skip test for bandfilter method -- different initial conditions do change results slightly at start
    # Skip test for multitaper phase/complex -- not time-reversal invariant
    if (method == 'wavelet') or ((method == 'multitaper') and (spec_type == 'power')):
        spec, freqs = spectrum(np.flip(data,axis=0), smp_rate, axis=0, method=method,
                               data_type=data_type, spec_type=spec_type)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert spec.shape == (n_freqs, n_trials)
        assert np.isclose(spec[:,0].mean(), reversed_result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        spec, freqs = spectrum(data, smp_rate, axis=0, method=method, data_type=data_type,
                            spec_type=spec_type, foo=None)


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
    data_orig = data.copy()
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
    else:                       dtype = float

    # Time reversal -> inverted sign phase, complex conj of complex, preserves power
    if spec_type == 'phase':        reversed_result = -result
    elif spec_type == 'complex':    reversed_result = np.conj(result)
    else:                           reversed_result = result

    # Basic test of shape, dtype, value of output.
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method,
                                       data_type=data_type_, spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts, n_trials)
    assert np.issubdtype(spec.dtype,dtype)
    assert np.isclose(spec[:,:,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs, timepts = spectrogram(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                                       smp_rate, axis=0, method=method, data_type=data_type_,
                                       spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts, n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,dtype)
    assert np.isclose(spec[:,:,0,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    extra_args = {'trial_axis':0} if spec_type == 'burst' else {}
    spec, freqs, timepts = spectrogram(data.T, smp_rate, axis=-1, method=method,
                                       data_type=data_type_, spec_type=spec_type, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
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
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
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
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert spec.shape == (n_freqs, n_timepts, n_trials)
        assert np.isclose(spec[:,:,0].mean(), reversed_result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method,
                                        data_type=data_type_, spec_type=spec_type, foo=None)


@pytest.mark.parametrize('itpc_method, method, result',
                         [('plv',   'wavelet',     0.5494),
                          ('plv',   'multitaper',  0.2705),
                          ('plv',   'bandfilter',  0.6556),
                          ('z',     'wavelet',     1.5316),
                          ('z',     'multitaper',  0.3762),
                          ('z',     'bandfilter',  2.0734),
                          ('ppc',   'wavelet',     0.1772),
                          ('ppc',   'multitaper',  -0.2079),
                          ('ppc',   'bandfilter',  0.3578)])
def test_itpc(oscillation, itpc_method, method, result):
    """ Unit tests for spectrogram() function """
    # Extract given data type from data dict
    data = oscillation
    data_orig = data.copy()
    smp_rate = 1000
    n_trials = 4

    method_to_n_freqs   = {'wavelet': 26, 'multitaper':257, 'bandfilter': 3, 'burst':4}
    method_to_n_timepts = {'wavelet': 1000, 'multitaper':2, 'bandfilter': 1000}
    n_freqs = method_to_n_freqs[method]
    freqs_shape = (n_freqs,2) if method == 'bandfilter' else (n_freqs,)
    n_timepts = method_to_n_timepts[method]

    # Basic test of shape, dtype, value of output.
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    spec, freqs, timepts = itpc(data, smp_rate, axis=0, method=method, itpc_method=itpc_method,
                                trial_axis=-1)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts)
    assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs, timepts = itpc(np.tile(data[:,:,np.newaxis],(1,1,2)), smp_rate, axis=0,
                                method=method, itpc_method=itpc_method, trial_axis=1)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts, 2)
    assert np.isclose(spec[:,:,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    spec, freqs, timepts = itpc(data.T, smp_rate, axis=-1, method=method, itpc_method=itpc_method,
                                trial_axis=0)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts)
    assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for expected output with time-reversed data
    # Skip test for bandfilter method -- different initial conditions do change results slightly at start
    # Skip test for multitaper method -- not time-reversal invariant due to windowing
    if method not in ['bandfilter','multitaper']:
        spec, freqs, timepts = itpc(np.flip(data,axis=0), smp_rate, axis=0, method=method,
                                    itpc_method=itpc_method, trial_axis=-1)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert spec.shape == (n_freqs, n_timepts)
        assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        spec, freqs, timepts = itpc(data, smp_rate, axis=0, method=method, itpc_method=itpc_method,
                                    trial_axis=-1, foo=None)


# =============================================================================
# Unit tests for preprocessing/postprocessing/utility functions
# =============================================================================
def test_cut_trials(oscillation):
    """ Unit tests for cut_trials function """
    n_timepts, n_trials = oscillation.shape

    # Unwrap trial-cut data into one long vector to simulate uncut data
    uncut_data  = oscillation.flatten(order='F')
    data        = oscillation
    data_orig   = uncut_data.copy()

    # Check if unwrapped->recut data is same as original uncut data
    trial_lims  = np.asarray([0,0.999])[np.newaxis,:] + np.arange(n_trials)[:,np.newaxis]
    cut_data    = cut_trials(uncut_data, trial_lims, smp_rate=1000, axis=0)
    assert np.array_equal(uncut_data,data_orig)     # Ensure input data isn't altered by function
    assert cut_data.shape == data.shape
    assert (cut_data == data).all()


def test_realign_data(oscillation):
    """ Unit tests for realign_data function """
    data = oscillation
    data_orig = data.copy()
    n_timepts, n_trials = data.shape
    timepts = np.arange(0,1,1e-3)

    # Realign to 2 distinct times, then concatenate together and test if same
    realigned1 = realign_data(data, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                timepts=timepts, time_axis=0, trial_axis=-1)
    realigned2 = realign_data(data, 0.5*np.ones((n_trials,)), time_range=(0,0.499),
                                timepts=timepts, time_axis=0, trial_axis=-1)
    realigned = np.concatenate((realigned1,realigned2), axis=0)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert realigned.shape == data.shape
    assert (realigned == data).all()

    # Test for consistent output with transposed data dimensionality
    realigned1 = realign_data(data.T, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                timepts=timepts, time_axis=-1, trial_axis=0)
    realigned2 = realign_data(data.T, 0.5*np.ones((n_trials,)), time_range=(0,0.499),
                                timepts=timepts, time_axis=-1, trial_axis=0)
    realigned = np.concatenate((realigned1,realigned2), axis=-1)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert realigned.shape == data.T.shape
    assert (realigned == data.T).all()


@pytest.mark.parametrize('variable_type, pooler, result',
                         [('numpy',   'mean',   43.5538),
                          ('numpy',   'sum',    300.0944),
                          ('numpy',   'custom', 43.5538),
                          ('xarray',  'mean',   43.5538),
                          ('xarray',  'sum',    300.0944),
                          ('xarray',  'custom', 43.5538)])
def test_pool_freq_bands(oscillation, variable_type, pooler, result):
    """ Unit tests for pool_freq_bands function """
    data = oscillation
    n_timepts, n_trials = data.shape
    smp_rate = 1000
    timepts = np.arange(0,1,1e-3)

    # Set 'func' = callable custom pooling function
    if pooler == 'custom': pooler = lambda x: np.mean(x,axis=0)

    variable_type_ = xr.DataArray if variable_type == 'xarray' else np.ndarray

    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method='wavelet',
                                       data_type='lfp', spec_type='power')
    # Convert Numpy ndarray -> xarray DataArray
    if variable_type == 'xarray':
        spec = xr.DataArray(spec,
                            dims=['frequency','time','trial'],
                            coords={'frequency':freqs, 'time':timepts, 'trial':np.arange(n_trials)})
    spec_orig = spec.copy()

    # Frequency bands to pool data within
    bands   = OrderedDict({'theta':[3,8], 'beta':[10,32], 'gamma':[40,100]})
    n_bands = len(bands)

    extra_args = dict(axis=0, freqs=freqs) if variable_type == 'numpy' else {}

    eval_func = (lambda x: x.mean().values) if variable_type == 'xarray' else (lambda x: x.mean())

    # Basic test of shape, type, value of output.
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    band_spec = pool_freq_bands(spec, bands, func=pooler, **extra_args)
    # print(band_spec[:,:,0].mean(), np.mean(band_spec[:,:,0]))
    # print(eval_func(band_spec[:,:,0]))
    assert np.array_equal(spec,spec_orig)     # Ensure input data isn't altered by function
    assert isinstance(band_spec,variable_type_)
    assert band_spec.shape == (n_bands, n_timepts, n_trials)
    assert np.isclose(eval_func(band_spec[:,:,0]), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with different data array shape (3rd axis)
    spec2 = np.tile(spec[:,:,:,np.newaxis],(1,1,2)) if variable_type == 'numpy' else \
            xr.concat((spec,spec), dim='channel').transpose('frequency','time','trial','channel')
    band_spec = pool_freq_bands(spec2, bands, func=pooler, **extra_args)
    assert np.array_equal(spec,spec_orig)     # Ensure input data isn't altered by function
    assert band_spec.shape == (n_bands, n_timepts, n_trials, 2)
    assert np.isclose(eval_func(band_spec[:,:,0,0]), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    if variable_type == 'numpy': extra_args['axis'] = 2
    band_spec = pool_freq_bands(spec.transpose(), bands, func=pooler, **extra_args)
    assert np.array_equal(spec,spec_orig)     # Ensure input data isn't altered by function
    assert band_spec.shape == (n_trials, n_timepts, n_bands)
    assert np.isclose(eval_func(band_spec[0,:,:]), result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        band_spec = pool_freq_bands(spec, bands, func=pooler, foo=None, **extra_args)


@pytest.mark.parametrize('variable_type, pooler, result',
                         [('numpy',   'mean',   44.9441),
                          ('numpy',   'sum',    22494.0313),
                          ('numpy',   'custom', 44.9441),
                          ('xarray',  'mean',   44.9441),
                          ('xarray',  'sum',    22494.0313),
                          ('xarray',  'custom', 44.9441)])
def test_pool_time_epochs(oscillation, variable_type, pooler, result):
    """ Unit tests for pool_time_epochs function """
    data = oscillation
    n_timepts, n_trials = data.shape
    smp_rate = 1000
    timepts = np.arange(0,1,1e-3)

    # Set 'func' = callable custom pooling function
    if pooler == 'custom': pooler = lambda x: np.mean(x,axis=0)

    variable_type_ = xr.DataArray if variable_type == 'xarray' else np.ndarray

    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method='wavelet',
                                       data_type='lfp', spec_type='power')
    # Convert Numpy ndarray -> xarray DataArray
    if variable_type == 'xarray':
        spec = xr.DataArray(spec,
                            dims=['frequency','time','trial'],
                            coords={'frequency':freqs, 'time':timepts, 'trial':np.arange(n_trials)})
    spec_orig = spec.copy()
    n_freqs = len(freqs)

    # Make up some time epochs to pool data within
    epochs   = OrderedDict({'sample':[0,0.5], 'delay':[0.5,1]})
    n_epochs = len(epochs)

    extra_args = dict(axis=1, timepts=timepts) if variable_type == 'numpy' else {}

    eval_func = (lambda x: x.mean().values) if variable_type == 'xarray' else (lambda x: x.mean())

    # Basic test of shape, type, value of output.
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    epoch_spec = pool_time_epochs(spec, epochs, func=pooler, **extra_args)
    print(epoch_spec[:,:,0].mean(), np.mean(epoch_spec[:,:,0]))
    print(eval_func(epoch_spec[:,:,0]))
    assert np.array_equal(spec,spec_orig)     # Ensure input data isn't altered by function
    assert isinstance(epoch_spec,variable_type_)
    assert epoch_spec.shape == (n_freqs, n_epochs, n_trials)
    assert np.isclose(eval_func(epoch_spec[:,:,0]), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with different data array shape (3rd axis)
    spec2 = np.tile(spec[:,:,:,np.newaxis],(1,1,2)) if variable_type == 'numpy' else \
            xr.concat((spec,spec), dim='channel').transpose('frequency','time','trial','channel')
    epoch_spec = pool_time_epochs(spec2, epochs, func=pooler, **extra_args)
    assert np.array_equal(spec,spec_orig)     # Ensure input data isn't altered by function
    assert epoch_spec.shape == (n_freqs, n_epochs, n_trials, 2)
    assert np.isclose(eval_func(epoch_spec[:,:,0,0]), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    if variable_type == 'numpy': extra_args['axis'] = 2
    spec2 = spec.transpose((0,2,1)) if variable_type == 'numpy' else \
            spec.transpose('frequency','trial','time')
    epoch_spec = pool_time_epochs(spec2, epochs, func=pooler, **extra_args)
    assert np.array_equal(spec,spec_orig)     # Ensure input data isn't altered by function
    assert epoch_spec.shape == (n_freqs, n_trials, n_epochs)
    assert np.isclose(eval_func(epoch_spec[:,0,:]), result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        epoch_spec = pool_time_epochs(spec, epochs, func=pooler, foo=None, **extra_args)
