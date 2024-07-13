""" Unit tests for spectra.py module """
from collections import OrderedDict
import pytest
import numpy as np
import xarray as xr

from spynal.utils import iarange
from spynal.tests.data_fixtures import oscillation, bursty_oscillation, spiking_oscillation, \
                                       oscillatory_data, MISSING_ARG_ERRS
from spynal.spectra.spectra import spectrum, spectrogram, itpc, plot_spectrum, plot_spectrogram
from spynal.spectra.preprocess import cut_trials, realign_data, remove_dc, remove_evoked
from spynal.spectra.postprocess import one_over_f_norm, pool_freq_bands, pool_time_epochs
from spynal.spectra.utils import get_freq_sampling,fft, ifft, one_sided_to_two_sided, power


# =============================================================================
# Unit tests for spectral analysis functions
# =============================================================================
@pytest.mark.parametrize('fft_method', ['torch','fftw','scipy','numpy'])
def test_fft(oscillation, fft_method):
    """ Unit tests for low-level FFT functions """
    n_fft = 1000
    n_trials = 4
    data = np.random.randn(n_fft,n_trials)
    data_orig = data.copy()

    # Basic test that ifft(fft(data)) = data
    spec = fft(data, n_fft=n_fft, axis=0, fft_method=fft_method)
    data2 = ifft(spec, n_fft=n_fft, axis=0, fft_method=fft_method)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.allclose(data, data2)

    # Test for consistent output with transposed data dimensionality
    spec = fft(data.T, n_fft=n_fft, axis=-1, fft_method=fft_method)
    data2 = ifft(spec, n_fft=n_fft, axis=-1, fft_method=fft_method)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.allclose(data.T, data2)

    # Test for consistent output with vector-valued data
    spec = fft(data[:,0], n_fft=n_fft, axis=0, fft_method=fft_method)
    data2 = ifft(spec, n_fft=n_fft, axis=0, fft_method=fft_method)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.allclose(data[:,0], data2)

    # Test for consistent output with different data array shape (3rd axis)
    data = data.reshape((-1,int(n_trials/2),int(n_trials/2)))
    data_orig = data.copy()
    spec = fft(data, n_fft=n_fft, axis=0, fft_method=fft_method)
    data2 = ifft(spec, n_fft=n_fft, axis=0, fft_method=fft_method)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.allclose(data, data2)

    # Test for consistent output with 3d data and axis=1
    data = data.transpose((1,0,2))
    spec = fft(data, n_fft=n_fft, axis=1, fft_method=fft_method)
    data2 = ifft(spec, n_fft=n_fft, axis=1, fft_method=fft_method)
    assert np.allclose(data, data2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        fft(data, n_fft=n_fft, axis=0, fft_method=fft_method, foo=None)
        ifft(data, n_fft=n_fft, axis=0, fft_method=fft_method, foo=None)


@pytest.mark.parametrize('data_type, spec_type, method, fft_method, result',
                         [('lfp',   'power', 'multitaper',  'torch',    0.0137),
                          ('lfp',   'power', 'multitaper',  'fftw',     0.0137),
                          ('lfp',   'power', 'multitaper',  'scipy',    0.0137),
                          ('lfp',   'power', 'multitaper',  'numpy',    0.0137),
                          ('lfp',   'power', 'wavelet',     'torch',    44.9462),
                          ('lfp',   'power', 'wavelet',     'fftw',     44.9462),
                          ('lfp',   'power', 'wavelet',     'scipy',    44.9462),
                          ('lfp',   'power', 'wavelet',     'numpy',    44.9462),
                          ('lfp',   'power', 'bandfilter',  None,       2.3097),
                          ('spike', 'power', 'multitaper',  'torch',    194.0514),
                          ('spike', 'power', 'multitaper',  'fftw',     194.0514),
                          ('spike', 'power', 'multitaper',  'scipy',    194.0514),
                          ('spike', 'power', 'multitaper',  'numpy',    194.0514),
                          ('spike', 'power', 'wavelet',     'torch',    0.4774),
                          ('spike', 'power', 'wavelet',     'fftw',     0.4774),
                          ('spike', 'power', 'wavelet',     'scipy',    0.4774),
                          ('spike', 'power', 'wavelet',     'numpy',    0.4774),
                          ('spike', 'power', 'bandfilter',  None,       0.0545)])
def test_spectrum(oscillatory_data, data_type, spec_type, method, fft_method, result):
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
                           fft_method=fft_method, spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs, n_trials)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec[:,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs = spectrum(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                           smp_rate, axis=0, method=method, data_type=data_type,
                           fft_method=fft_method, spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs, n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec[:,0,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    spec, freqs = spectrum(data.T, smp_rate, axis=-1, method=method, data_type=data_type,
                           fft_method=fft_method, spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_trials, n_freqs)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec[0,:].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    spec, freqs = spectrum(data[:,0], smp_rate, axis=-1, method=method, data_type=data_type,
                           fft_method=fft_method, spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert spec.shape == (n_freqs,)
    assert np.issubdtype(spec.dtype,float)
    assert np.isclose(spec.mean(), result, rtol=1e-4, atol=1e-4)

    # Test for expected output with time-reversed data
    # Skip test for bandfilter method -- different init conds do change results slightly at start
    # Skip test for multitaper phase/complex -- not time-reversal invariant
    if (method == 'wavelet') or ((method == 'multitaper') and (spec_type == 'power')):
        spec, freqs = spectrum(np.flip(data,axis=0), smp_rate, axis=0, method=method,
                               fft_method=fft_method, data_type=data_type, spec_type=spec_type)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert spec.shape == (n_freqs, n_trials)
        assert np.isclose(spec[:,0].mean(), reversed_result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        spec, freqs = spectrum(data, smp_rate, axis=0, method=method, data_type=data_type,
                               fft_method=fft_method, spec_type=spec_type, foo=None)


# TODO Take a closer look at 'complex' outputs -- why coming out = 0?
@pytest.mark.parametrize('data_type, spec_type, method, fft_method, result',
                         [('lfp',   'power',    'wavelet',     'torch', 44.9462),
                          ('lfp',   'power',    'wavelet',     'fftw',  44.9462),
                          ('lfp',   'power',    'wavelet',     'scipy', 44.9462),
                          ('lfp',   'power',    'wavelet',     'numpy', 44.9462),
                          ('lfp',   'power',    'multitaper',  'torch', 0.0137),
                          ('lfp',   'power',    'multitaper',  'fftw', 0.0137),
                          ('lfp',   'power',    'multitaper',  'scipy', 0.0137),
                          ('lfp',   'power',    'multitaper',  'numpy', 0.0137),
                          ('lfp',   'power',    'bandfilter',  None,    2.3097),
                          ('burst', 'burst',    'wavelet',     None,    0.0),
                          ('burst', 'burst',    'bandfilter',  None,    0.0),
                          ('spike', 'power',    'wavelet',     None,    0.4774),
                          ('spike', 'power',    'multitaper',  None,    193.7612),
                          ('spike', 'power',    'bandfilter',  None,    0.0545),
                          ('lfp',   'phase',    'wavelet',     'torch', 0.0053),
                          ('lfp',   'phase',    'wavelet',     'fftw', 0.0053),
                          ('lfp',   'phase',    'wavelet',     'scipy', 0.0053),
                          ('lfp',   'phase',    'wavelet',     'numpy', 0.0053),
                          ('lfp',   'phase',    'multitaper', 'torch',  -0.1523),
                          ('lfp',   'phase',    'multitaper', 'scipy',  -0.1523),
                          ('lfp',   'phase',    'multitaper', 'scipy',  -0.1523),
                          ('lfp',   'phase',    'multitaper', 'numpy',  -0.1523),
                          ('lfp',   'phase',    'bandfilter',  None,    0.0799),
                          ('spike', 'phase',    'wavelet',     None,    0.0054),
                          ('spike', 'phase',    'multitaper',  None,    -0.0547),
                          ('spike', 'phase',    'bandfilter',  None,    -0.0764),
                          ('lfp',   'complex',  'wavelet',     None,    0j),
                          ('lfp',   'complex',  'multitaper',  None,    0.0019-0.0011j),
                          ('lfp',   'complex',  'bandfilter',  None,    -0.0063+0j)])
def test_spectrogram(oscillatory_data, data_type, spec_type, method, fft_method, result):
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
    if spec_type == 'complex':  dtype = complex
    elif spec_type == 'burst':  dtype = bool
    else:                       dtype = float

    # Time reversal -> inverted sign phase, complex conj of complex, preserves power
    if spec_type == 'phase':        reversed_result = -result
    elif spec_type == 'complex':    reversed_result = np.conj(result)
    else:                           reversed_result = result

    # Basic test of shape, dtype, value of output.
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method,
                                       data_type=data_type_, fft_method=fft_method,
                                       spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts, n_trials)
    assert np.issubdtype(spec.dtype,dtype)
    assert np.isclose(spec[:,:,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with different data array shape (3rd axis)
    spec, freqs, timepts = spectrogram(data.reshape((-1,int(n_trials/2),int(n_trials/2))),
                                       smp_rate, axis=0, method=method, data_type=data_type_,
                                       fft_method=fft_method, spec_type=spec_type)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert freqs.shape == freqs_shape
    assert timepts.shape == (n_timepts,)
    assert spec.shape == (n_freqs, n_timepts, n_trials/2, n_trials/2)
    assert np.issubdtype(spec.dtype,dtype)
    assert np.isclose(spec[:,:,0,0].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    extra_args = {'trial_axis':0} if spec_type == 'burst' else {}
    spec, freqs, timepts = spectrogram(data.T, smp_rate, axis=-1, method=method,
                                       data_type=data_type_, fft_method=fft_method,
                                       spec_type=spec_type, **extra_args)
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
                                           data_type=data_type_, fft_method=fft_method,
                                           spec_type=spec_type)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert freqs.shape == freqs_shape
        assert timepts.shape == (n_timepts,)
        assert spec.shape == (n_freqs, n_timepts)
        assert np.issubdtype(spec.dtype,dtype)
        assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)

    # Test for expected output with time-reversed data
    # Skip test for bandfilter method -- different init conds do change results slightly at start
    # Skip test for multitaper phase/complex -- not time-reversal invariant
    if (method == 'wavelet') or ((method == 'multitaper') and (spec_type == 'power')):
        spec, freqs, timepts = spectrogram(np.flip(data,axis=0), smp_rate, axis=0, method=method,
                                           data_type=data_type_, fft_method=fft_method,
                                           spec_type=spec_type)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert spec.shape == (n_freqs, n_timepts, n_trials)
        assert np.isclose(spec[:,:,0].mean(), reversed_result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method,
                                           data_type=data_type_, fft_method=fft_method,
                                           spec_type=spec_type, foo=None)


@pytest.mark.parametrize('itpc_method, method, result',
                         [('plv',   'wavelet',     0.5494),
                          ('plv',   'multitaper',  0.2705),
                          ('plv',   'bandfilter',  0.7836),
                          ('z',     'wavelet',     1.5316),
                          ('z',     'multitaper',  0.3762),
                          ('z',     'bandfilter',  2.7367),
                          ('ppc',   'wavelet',     0.1772),
                          ('ppc',   'multitaper',  -0.2079),
                          ('ppc',   'bandfilter',  0.5789)])
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
    # Skip test for bandfilter method -- different init conds do change results slightly at start
    # Skip test for multitaper method -- not time-reversal invariant due to windowing
    if method not in ['bandfilter','multitaper']:
        spec, freqs, timepts = itpc(np.flip(data,axis=0), smp_rate, axis=0, method=method,
                                    itpc_method=itpc_method, trial_axis=-1)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert spec.shape == (n_freqs, n_timepts)
        assert np.isclose(spec[:,:].mean(), result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        spec, freqs, timepts = itpc(data, smp_rate, axis=0, method=method, itpc_method=itpc_method,
                                    trial_axis=-1, foo=None)


# =============================================================================
# Unit tests for spectral/LFP preprocessing functions
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

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        cut_data = cut_trials(uncut_data, trial_lims, smp_rate=1000, axis=0, foo=None)


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

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        realigned = realign_data(data, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                 timepts=timepts, time_axis=0, trial_axis=-1, foo=None)


@pytest.mark.parametrize('axis', [0, None])
def test_remove_dc(oscillation, axis):
    """ Unit tests for remove_dc() function """
    data = oscillation
    data_orig = data.copy()

    # Test that DC-removed data does have mean=0
    data_no_dc = remove_dc(data, axis=axis)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data_no_dc.shape, data.shape)
    if axis is None:    assert np.isclose(data_no_dc.sum(), 0)
    else:               assert np.allclose(data_no_dc.sum(axis=axis), 0)

    # Test for consistency with transposed data
    axis_T = 1 if axis == 0 else None
    data_no_dc = remove_dc(data.T, axis=axis_T)
    assert np.array_equal(data_no_dc.shape, data.T.shape)
    if axis is None:    assert np.isclose(data_no_dc.sum(), 0)
    else:               assert np.allclose(data_no_dc.sum(axis=axis_T), 0)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        data_no_dc = remove_dc(data, axis=axis, foo=None)


@pytest.mark.parametrize('method, result, result2',
                         [('mean',      13.3, 0.31),
                          ('groupmean', 0.49, 13.13),
                          ('regress',   0.49, 13.13)])
def test_remove_evoked(oscillation, method, result, result2):
    """ Unit tests for remove_evoked() function """
    data = oscillation
    data[:,2:] = np.roll(data[:,2:], shift=15, axis=0)  # Shift last 2 simulated trials by ~180 deg
    data_orig = data.copy()

    if method == 'mean':        design = None
    elif method == 'groupmean': design = [0,0,1,1]
    elif method == 'regress':   design = np.vstack(([0,0,0,0], [-1,-1,1,1])).T

    # Basic test of function
    data_no_evoked, evoked = remove_evoked(data, axis=1, method=method, design=design, return_evoked=True)
    print(np.round(power(data_no_evoked).mean(),2), np.round(power(evoked).mean(),2))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data_no_evoked.shape, data.shape)
    assert np.array_equal(evoked.shape, data.shape)
    assert np.isclose(power(data_no_evoked).mean(), result, rtol=1e-2, atol=1e-2)
    assert np.isclose(power(evoked).mean(), result2, rtol=1e-2, atol=1e-2)

    # Test for consistency with not returning evoked potential
    data_no_evoked = remove_evoked(data, axis=1, method=method, design=design, return_evoked=False)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data_no_evoked.shape, data.shape)
    assert np.isclose(power(data_no_evoked).mean(), result, rtol=1e-2, atol=1e-2)

    # Test for consistency with transposed data
    data_no_evoked = remove_evoked(data.T, axis=0, method=method, design=design)
    assert np.array_equal(data_no_evoked.shape, data.T.shape)
    assert np.isclose(power(data_no_evoked).mean(), result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        data_no_evoked = remove_evoked(data, axis=0, method=method, foo=None)


# =============================================================================
# Unit tests for spectral/LFP postprocessing functions
# =============================================================================
def test_one_over_f_norm():
    """ Unit tests for one_over_f_norm() function """
    # Generate idealized 1/f spectrogam ~ (n_freqs,n_trials) = (100,4)
    f = iarange(1,100)
    data = np.tile((1.0/f)[:,np.newaxis], (1,4))
    data_orig = data.copy()

    # Basic test of function
    data_corrected = one_over_f_norm(data, axis=0, freqs=f, exponent=1.0)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data_corrected.shape, data.shape)
    assert np.allclose(data_corrected.std(axis=0), 0)

    # Test for consistency for transposed data
    data_corrected = one_over_f_norm(data.T, axis=1, freqs=f, exponent=1.0)
    assert np.array_equal(data_corrected.shape, data.T.shape)
    assert np.allclose(data_corrected.std(axis=1), 0)

    # Test for expected output for higher exponent
    data = np.tile((1.0/f**2)[:,np.newaxis], (1,4))
    data_corrected = one_over_f_norm(data, axis=0, freqs=f, exponent=2.0)
    assert np.allclose(data_corrected.std(axis=0), 0)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        data_corrected = one_over_f_norm(data, axis=0, freqs=f, foo=None)


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
    with pytest.raises(MISSING_ARG_ERRS):
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
    with pytest.raises(MISSING_ARG_ERRS):
        epoch_spec = pool_time_epochs(spec, epochs, func=pooler, foo=None, **extra_args)


# =============================================================================
# Unit tests for spectral plotting functions
# =============================================================================
@pytest.mark.parametrize('method', [('multitaper'), ('wavelet'), ('bandfilter')])
def test_plot_spectrum(oscillation, method):
    """ Unit tests for plot_spectrum function with different frequency sampling """
    data = oscillation
    smp_rate = 1000

    spec, freqs = spectrum(data, smp_rate, axis=0, method=method, spec_type='power')
    mean = spec.mean(axis=-1)
    mean_orig = mean.copy()
    freqs_orig = freqs.copy()
    if method == 'wavelet':         freqs_result = np.log2(freqs)
    elif method == 'bandfilter':    freqs_result = np.arange(len(freqs))
    else:                           freqs_result = freqs

    # Basic test that plotted data == input data
    lines, _ = plot_spectrum(freqs, mean)
    assert np.array_equal(freqs, freqs_orig) # Ensure input data isn't altered by function
    assert np.array_equal(mean, mean_orig)
    assert np.allclose(lines[0].get_xdata(), freqs_result)
    assert np.allclose(lines[0].get_ydata(), mean)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        lines, _ = plot_spectrum(freqs, mean, foo=None)


@pytest.mark.parametrize('method', [('multitaper'), ('wavelet'), ('bandfilter')])
def test_plot_spectrogram(oscillation, method):
    """ Unit tests for plot_spectrogram function with different frequency sampling """
    data = oscillation
    smp_rate = 1000

    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method=method, spec_type='power')
    spec = spec.mean(axis=-1)
    freqs_orig = freqs.copy()
    spec_orig = spec.copy()

    # Basic test that plotted data == input data
    img, _ = plot_spectrogram(timepts, freqs, spec)
    assert np.array_equal(freqs, freqs_orig) # Ensure input data isn't altered by function
    assert np.array_equal(spec, spec_orig)
    assert np.allclose(img.get_array().data, spec)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        img, _ = plot_spectrogram(timepts, freqs, spec, foo=None)


# =============================================================================
# Unit tests for other spectral/LFP utility functions
# =============================================================================
def test_get_freq_sampling():
    """ Unit tests for get_freq_sampling() function """
    smp_rate = 1000
    nfft = 1000

    # Basic test of function
    f, fbool = get_freq_sampling(smp_rate, nfft)
    print(f.shape, f[:10], f[-1])
    assert np.array_equal(f, iarange(0,500))

    # Test with different freq_range, nfft, smp_rate
    f, fbool = get_freq_sampling(smp_rate, nfft, freq_range=(100,200))
    assert np.array_equal(f, iarange(100,200))

    f, fbool = get_freq_sampling(smp_rate, 500)
    assert np.array_equal(f, iarange(0,500,2))

    f, fbool = get_freq_sampling(smp_rate, nfft, two_sided=True)
    assert np.array_equal(f, np.hstack((iarange(0,499), iarange(-500,-1))))

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        f, fbool = get_freq_sampling(smp_rate, nfft, foo=None)


def test_one_sided_to_two_sided(oscillation):
    """ Unit tests for one_sided_to_two_sided() function """
    data = oscillation
    smp_rate = 1000
    n_trials = 4

    # Basic test of shape, dtype, value of output.
    # Test values averaged over all timepts, freqs for 1st trial for simplicity
    spec, freqs, timepts = spectrogram(data, smp_rate, axis=0, method='multitaper',
                                       data_type='lfp', spec_type='complex')
    spec_orig = spec.copy()

    # TODO Set up some basic tests of function
    _ = one_sided_to_two_sided(spec, freqs, smp_rate, axis=0)
    assert np.array_equal(spec,spec_orig)     # Ensure input data isn't altered by function

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = one_sided_to_two_sided(spec, freqs, smp_rate, axis=0, foo=None)


def test_imports():
    """ Test different import methods for spectra module """
    # Import entire package
    import spynal
    spynal.spectra.wavelet.wavelet_spectrogram
    spynal.spectra.wavelet_spectrogram
    # Import module
    import spynal.spectra as spec
    spec.wavelet.wavelet_spectrogram
    spec.wavelet_spectrogram
    # Import specific function from module
    from spynal.spectra import wavelet_spectrogram
    wavelet_spectrogram
    # Import specific function from module
    from spynal.spectra.wavelet import wavelet_spectrogram
    wavelet_spectrogram
