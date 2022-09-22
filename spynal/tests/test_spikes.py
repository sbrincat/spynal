""" Unit tests for spikes.py module """
import pytest
import numpy as np

from spynal.tests.data_fixtures import MISSING_ARG_ERRS
from spynal.utils import setup_sliding_windows, unsorted_unique, \
                         object_array_equal, concatenate_object_array
from spynal.spikes import simulate_spike_trains, simulate_spike_waveforms, \
                          times_to_bool, bool_to_times, \
                          cut_trials, realign_data, pool_electrode_units, \
                          rate, rate_stats, isi, isi_stats, waveform_stats, \
                          plot_mean_waveforms, plot_waveform_heatmap
from spynal.spectra.multitaper import compute_tapers


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

    timepts None. Second argout returned only to match output of spike_bool
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

    SOURCE      https://stackoverflow.com/a/42400786
    """
    return {'spike_timestamp': spike_timestamp, 'spike_bool': spike_bool}


@pytest.fixture(scope='session')
def spike_timestamp_trial_uncut(spike_timestamp):
    """
    Simulates set of spike trains as spike timestamps not cut into trials

    RETURNS
    data    (2,) ndarray of (n_spikes,) objects. Simulated spike timestamps
            (eg simulating 2 units, uncut into trials)

    timepts None. Second argout returned only to match output of spike_bool
    """
    data, _ = spike_timestamp
    n_trials,n_units = data.shape
    data_uncut = np.empty((n_units,),dtype=object)
    for unit in range(n_units):
        data_uncut[unit] = np.hstack([data[trial,unit]+trial for trial in range(n_trials)])

    return data_uncut, None


@pytest.fixture(scope='session')
def spike_bool_trial_uncut(spike_bool):
    """
    Fixture simulates set of spike trains as binary trains to use for all unit tests

    RETURNS
    data    (2,1001) ndarray of bool. Simulated binary spike trains
            (eg simulating 2 units x 1001 timepts, uncut into trials).

    timepts (1001,) ndarray of float. Time sampling vector for data (in s).
    """
    data, timepts = spike_bool
    n_trials,n_units,n_timepts = data.shape
    data = data.transpose((1,2,0)).reshape((n_units,n_timepts*n_trials),order='F')
    timepts = np.hstack([timepts + trial*n_timepts for trial in range(n_trials)])
    return data, timepts


@pytest.fixture(scope='session')
def spike_data_trial_uncut(spike_timestamp_trial_uncut, spike_bool_trial_uncut):
    """
    "Meta-fixture" that returns both timestamp and boolean spiking data types
    in a dictionary.

    RETURNS
    data_dict   {'data_type' : (data,timepts)} dict containing outputs from
                each of constituent fixtures

    SOURCE      https://stackoverflow.com/a/42400786
    """
    return {'spike_timestamp': spike_timestamp_trial_uncut,
            'spike_bool': spike_bool_trial_uncut}


@pytest.fixture(scope='session')
def spike_waveform(spike_timestamp):
    """
    Fixture simulates spike waveforms of set of spike trains to use for all unit tests

    RETURNS
    data    (10,2) ndarray of (n_timepts,n_spikes) objects. Simulated spike waveforms
            (eg simulating 10 trials x 2 units)
            
    timepts (49,) ndarray of float. Time sampling vector for waveform data (in s).            
    """
    spike_timestamp = spike_timestamp[0]
    n_trials,n_units = spike_timestamp.shape
    
    spike_waves = np.empty((n_trials,n_units), dtype=object)
    for i_trial in range(n_trials):
        for i_unit in range(n_units):
            n_spikes = len(spike_timestamp[i_trial,i_unit])
            if n_spikes != 0:
                spike_waves[i_trial,i_unit],timepts = simulate_spike_waveforms(n_spikes=n_spikes)
            else:
                spike_waves[i_trial,i_unit] = np.asarray([])
            
    return spike_waves, timepts


# =============================================================================
# Unit tests for rate computation functions
# =============================================================================
@pytest.mark.parametrize('data_type, output, result',
                         [('spike_timestamp', 'count', 105),
                          ('spike_timestamp', 'rate', 5.25),
                          ('spike_bool', 'count', 105),
                          ('spike_bool', 'rate', 5.25)])
def test_bin_rate(spike_data, data_type, output, result):
    """ Unit tests for bin_rate function for computing binned spike rates """
    # Extract given data type from data dict
    data, timepts = spike_data[data_type]
    data_orig = data.copy()

    if output == 'rate':    dtype_type = float
    elif output == 'count': dtype_type = np.integer
    else:                   dtype_type = bool

    data_checker = np.array_equal if data_type == 'bool' else object_array_equal
    result_checker = np.mean if output == 'rate' else np.sum

    # Basic test of shape, dtype, value of output
    # Test values summed over entire array -> scalar for spike counts
    # Test values averaged over entire array -> scalar for spike rates
    rates, bins = rate(data, method='bin', lims=[0,1], output=output, axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert bins.shape == (20, 2)
    assert rates.shape == (10, 2, 20)
    assert np.issubdtype(rates.dtype,dtype_type)
    assert np.isclose(result_checker(rates), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different data array shape
    shape = (5,2,2,*data.shape[2:])
    rates, bins = rate(data.reshape(shape), method='bin', lims=[0,1], output=output,
                       axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert bins.shape == (20, 2)
    assert rates.shape == (5, 2, 2, 20)
    assert np.isclose(result_checker(rates), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    # Note: output dims are expected to be different for timestamp vs boolean data
    expected_shape = (20, 2, 10) if data_type == 'spike_bool' else (2, 10, 20)
    rates, bins = rate(data.transpose(), method='bin', lims=[0,1], output=output,
                       axis=0, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert bins.shape == (20, 2)
    assert rates.shape == expected_shape
    assert np.isclose(result_checker(rates), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different sliding window length
    rates, bins = rate(data, method='bin', lims=[0,1], width=20e-3, output=output,
                       axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert bins.shape == (50, 2)
    assert rates.shape == (10, 2, 50)
    assert np.isclose(result_checker(rates), result, rtol=1e-2, atol=1e-2)

    # Test for consistent ouptut when bins are set explicitly
    bins = setup_sliding_windows(20e-3,[0,1])
    rates, bins = rate(data, method='bin', bins=bins, output=output, axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert bins.shape == (50, 2)
    assert rates.shape == (10, 2, 50)
    assert np.isclose(result_checker(rates), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output for custom unequal-width bins
    # Note: Summed counts should be same, but mean rates are expected to be slightly different here
    bins = [[0,250e-3], [250e-3,750e-3], [750e-3,1000e-3]]
    rates, bins = rate(data, method='bin', bins=bins, output=output, axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert rates.shape == (10, 2, 3)
    if output == 'rate':
        assert np.isclose(rates.mean(), 5.53, rtol=1e-2, atol=1e-2)
    else:
        assert np.isclose(result_checker(rates), result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        rates, bins = rate(data, method='bin', lims=[0,1], output=output, axis=-1,
                           timepts=timepts, foo=None)


@pytest.mark.parametrize('data_type, kernel, result',
                         [('spike_timestamp', 'gaussian', 4.92),
                          ('spike_timestamp', 'hanning', 4.93),
                          ('spike_bool', 'gaussian', 5.25),
                          ('spike_bool', 'hanning', 5.25)])
def test_density(spike_data, data_type, kernel, result):
    """ Unit tests for density function for computing spike densities """
    # Extract given data type from data dict
    data, timepts = spike_data[data_type]
    data_orig = data.copy()

    # Set kernel width parameter so gaussian and hanning kernels are ~ identical
    width = 50e-3 if kernel == 'gaussian' else 125e-3

    data_checker = np.array_equal if data_type == 'bool' else object_array_equal

    # Basic test of shape, dtype, value of output
    # Test values averaged over entire array -> scalar for spike rates
    rates, tout = rate(data, method='density', kernel=kernel, width=width, lims=[0,1],
                       axis=-1, timepts=timepts)
    print(rates.mean(), tout.shape, rates.shape)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert tout.shape == (1001,)
    assert rates.shape == (10, 2, 1001)
    assert np.isclose(rates.mean(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different data array shape
    shape = (5,2,2,*data.shape[2:])
    rates, tout = rate(data.reshape(shape), method='density', kernel=kernel, width=width,
                       lims=[0,1], axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert tout.shape == (1001,)
    assert rates.shape == (5, 2, 2, 1001)
    assert np.isclose(rates.mean(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    # Note: output dims are expected to be different for timestamp vs boolean data
    expected_shape = (1001, 2, 10) if data_type == 'spike_bool' else (2, 10, 1001)
    print(data.shape, data.transpose().shape)
    rates, tout = rate(data.transpose(), method='density', kernel=kernel, width=width,
                       lims=[0,1], axis=0, timepts=timepts)
    print(rates.mean(), tout.shape, rates.shape)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert tout.shape == (1001,)
    assert rates.shape == expected_shape
    assert np.isclose(rates.mean(), result, rtol=1e-2, atol=1e-2)

    # Test for ~ consistent ouptut with 10x downsampling after spike density estimation
    rates, tout = rate(data, method='density', kernel=kernel, width=width, lims=[0,1],
                       step=10e-3, axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert tout.shape == (101,)
    assert rates.shape == (10, 2, 101)
    assert np.isclose(rates.mean(), result, rtol=0.01, atol=0.01)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        rates, tout = rate(data, method='density', kernel=kernel, width=width, lims=[0,1],
                           axis=-1, timepts=timepts, foo=None)


# =============================================================================
# Unit tests for spike rate, ISI, and waveform stats functions
# =============================================================================
@pytest.mark.parametrize('data_type, stat, result',
                         [('spike_timestamp', 'Fano', 42.0),
                          ('spike_bool', 'CV', 2.29)])
def test_rate_stats(spike_data, data_type, stat, result):
    """ Unit tests for rate_stats function for computing spike rate statistics """
    # Extract given data type from data dict
    data, timepts = spike_data[data_type]

    # Compute spike rates from timestamp/binary spike data -> (n_trials,n_unit,n_timepts)
    rates, _ = rate(data, method='bin', lims=[0,1], output='rate', axis=-1, timepts=timepts)
    rates_orig = rates.copy()
    n_trials,n_chnls,n_timepts = rates.shape

    data_checker = np.array_equal if data_type == 'bool' else object_array_equal

    # Basic test of shape, value of output
    # Test value of 1st trial/channel as exemplar
    stats = rate_stats(rates, stat=stat, axis=0)
    assert data_checker(rates,rates_orig)     # Ensure input data not altered by func
    assert stats.shape == (1, n_chnls, n_timepts)
    assert np.isclose(stats[0,0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different data array shape (additional axis inserted)
    stats = rate_stats(np.concatenate((rates[:,:,np.newaxis,:],rates[:,:,np.newaxis,:]), axis=2),
                       stat=stat, axis=0)
    assert data_checker(rates,rates_orig)     # Ensure input data not altered by func
    assert stats.shape == (1, n_chnls, 2, n_timepts)
    assert np.isclose(stats[0,0,0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    stats = rate_stats(rates.T, stat=stat, axis=-1)
    assert data_checker(rates,rates_orig)     # Ensure input data not altered by func
    assert stats.shape == (n_timepts, n_chnls, 1)
    assert np.isclose(stats[0,0,0], result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        stats = rate_stats(rates, stat=stat, axis=0, foo=None)


@pytest.mark.parametrize('data_type, stat, result',
                         [('spike_timestamp', 'Fano', 0.1006),
                          ('spike_bool', 'CV', 0.9110),
                          ('spike_bool', 'CV2', 0.9867),
                          ('spike_bool', 'LV', 0.9423),
                          ('spike_bool', 'burst_fract', 0.2041)])
def test_isi_stats(spike_data, data_type, stat, result):
    """ Unit tests for isi_stats function for computing inter-spike interval statistics """
    # Extract given data type from data dict
    data, timepts = spike_data[data_type]
    data_orig = data.copy()

    data_checker = np.array_equal if data_type == 'bool' else object_array_equal

    # Compute inter-spike intervals from timestamp/binary spike data -> (n_trials,n_units)
    ISIs = isi(data, axis=-1, timepts=timepts)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    n_trials,n_chnls = ISIs.shape

    # KLUDGE Pool ISI data across trials for locality-sensitive metrics, so output shape is same
    if stat in ['CV2','LV']: ISIs = concatenate_object_array(ISIs,axis=0,sort=False)
    ISIs_orig = ISIs.copy()

    axis = 'each' if stat in ['CV2','LV'] else 0

    # Basic test of shape, value of output
    # Test value of 1st trial/channel as exemplar
    stats = isi_stats(ISIs, stat=stat, axis=axis)
    assert data_checker(ISIs,ISIs_orig)     # Ensure input data not altered by func
    assert stats.shape == (1, n_chnls)
    assert np.isclose(stats[0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different data array shape (additional axis inserted)
    stats = isi_stats(np.concatenate((ISIs[:,:,np.newaxis],ISIs[:,:,np.newaxis]), axis=2),
                      stat=stat, axis=axis)
    assert data_checker(ISIs,ISIs_orig)     # Ensure input data not altered by func
    assert stats.shape == (1, n_chnls, 2)
    assert np.isclose(stats[0,0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    stats = isi_stats(ISIs.T, stat=stat, axis='each' if stat in ['CV2','LV'] else -1)
    assert data_checker(ISIs,ISIs_orig)     # Ensure input data not altered by func
    assert stats.shape == (n_chnls, 1)
    assert np.isclose(stats[0,0], result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        ISIs = isi(data, axis=-1, timepts=timepts, foo=None)
        stats = isi_stats(ISIs, stat=stat, axis=axis, foo=None)


@pytest.mark.parametrize('stat, result',
                         [('width',         433.33),
                          ('repolarization',200.00),
                          ('trough_width',  166.67),
                          ('amp_ratio',     1.55)])
def test_waveform_stats(spike_waveform, stat, result):
    """ Unit tests for waveform_stats function for computing spike waveform statistics """
    # Extract given data type from data dict
    data,timepts = spike_waveform
    n_trials,n_units = data.shape
    n_timepts,n_spikes0 = data[0,0].shape
    data_orig = data.copy()

    extra_args = dict(smp_rate=30e3) if stat != 'amp_ratio' else {}
    result_mult = 1 if stat == 'amp_ratio' else 1e6 # Convert temporal stats from s -> us
    
    # Basic test of shape, value of output
    # Test value of 1st trial/channel as exemplar
    stats = waveform_stats(data, stat=stat, axis=0, **extra_args)
    print(stats[0,0].shape, np.round(stats[0,0][0,0]*result_mult,2))
    assert object_array_equal(data,data_orig)     # Ensure input data not altered by func
    assert stats.shape == (n_trials,n_units)
    assert stats[0,0].shape == (1,n_spikes0)
    assert np.isclose(stats[0,0][0,0]*result_mult, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with non-object array inputs (just ndarray for single trial,unit)
    trial_data = data[0,0]
    trial_data_orig = trial_data.copy()
    stats = waveform_stats(trial_data, stat=stat, axis=0, **extra_args)
    assert np.array_equal(trial_data,trial_data_orig)     # Ensure input data not altered by func
    assert stats.shape == (1, n_spikes0)
    assert np.isclose(stats[0,0]*result_mult, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    stats = waveform_stats(trial_data.T, stat=stat, axis=-1, **extra_args)
    assert np.array_equal(trial_data,trial_data_orig)     # Ensure input data not altered by func
    assert stats.shape == (n_spikes0, 1)
    assert np.isclose(stats[0,0]*result_mult, result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        stats = waveform_stats(data, stat=stat, axis=0, foo=None)

  
# =============================================================================
# Unit tests for rate preprocessing/utility functions
# =============================================================================
def test_bool_to_times(spike_timestamp, spike_bool):
    """ Unit tests for bool_to_times function to convert spike timestamps to binary trains """
    data_timestamp, _   = spike_timestamp
    data_bool, timepts  = spike_bool
    data_orig = data_bool.copy()

    # Test that timestamp->bool->timestamp data retains same dtype, shape, values
    data_bool_to_timestamp = bool_to_times(data_bool, timepts, axis=-1)
    assert np.array_equal(data_bool,data_orig)  # Ensure input data not altered by func
    assert data_bool_to_timestamp.dtype == data_timestamp.dtype
    assert data_bool_to_timestamp.shape == data_timestamp.shape
    assert np.asarray([d1.shape == d2.shape for d1,d2
                       in zip(data_bool_to_timestamp.flatten(), data_timestamp.flatten())]).all()
    assert np.asarray([np.allclose(d1, d2, rtol=1e-2, atol=1e-2) for d1,d2
                       in zip(data_bool_to_timestamp.flatten(), data_timestamp.flatten())]).all()

    # Test for correct handling of single spike trains
    assert np.allclose(bool_to_times(data_bool[0,0,:], timepts, axis=-1), data_timestamp[0,0],
                       rtol=1e-2, atol=1e-2)
    assert np.array_equal(data_bool,data_orig)  # Ensure input data not altered by func


def test_times_to_bool(spike_timestamp, spike_bool):
    """ Unit tests for times_to_bool function to convert binary spike trains to timestamps """
    data_timestamp, _   = spike_timestamp
    data_bool, timepts  = spike_bool
    data_orig = data_timestamp.copy()

    # Test that bool->timestamp->bool data retains same shape, dtype, values
    data_timestamp_to_bool, timepts2 = times_to_bool(data_timestamp, lims=(0,1))
    assert object_array_equal(data_timestamp,data_orig)  # Ensure input data not altered by func
    assert (timepts == timepts2).all()
    assert data_timestamp_to_bool.shape == data_bool.shape
    assert data_timestamp_to_bool.dtype == data_bool.dtype
    assert (data_timestamp_to_bool == data_bool).all()

    # Test for correct handling of single spike trains and list-valued data
    assert (times_to_bool(data_timestamp[0,0], lims=(0,1))[0] == data_bool[0,0,:]).all()
    assert object_array_equal(data_timestamp,data_orig)  # Ensure input data not altered by func
    assert (times_to_bool(list(data_timestamp[0,0]), lims=(0,1))[0] == data_bool[0,0,:]).all()
    assert object_array_equal(data_timestamp,data_orig)  # Ensure input data not altered by func


@pytest.mark.parametrize('data_type',
                         [('spike_timestamp'),('spike_bool')])
def test_cut_trials(spike_data_trial_uncut, spike_data, data_type):
    """ Unit tests for cut_trials function """
    data, _ = spike_data[data_type]
    uncut_data, _ = spike_data_trial_uncut[data_type]
    data_orig = uncut_data.copy()
    n_trials, n_units = data.shape[:2]

    if data_type == 'spike_timestamp':
        trial_lims = np.asarray([0,1])[np.newaxis,:] + np.arange(n_trials)[:,np.newaxis]
        cut_data = cut_trials(uncut_data, trial_lims, trial_refs=np.arange(0,n_trials)).T
        assert object_array_equal(uncut_data,data_orig)     # Ensure input data not altered by func
        for trial in range(n_trials):
            for unit in range(n_units):
                assert np.allclose(cut_data[trial,unit], data[trial,unit])

        # Ensure that passing a nonexistent/misspelled kwarg raises an error
        with pytest.raises(MISSING_ARG_ERRS):
            cut_data = cut_trials(uncut_data, trial_lims, trial_refs=np.arange(0,n_trials), foo=None).T

    else:
        trial_lims = np.asarray([0,1])[np.newaxis,:] + 1.001*np.arange(n_trials)[:,np.newaxis]
        cut_data = cut_trials(uncut_data, trial_lims, smp_rate=1000, axis=1).transpose((2,0,1))
        assert np.array_equal(uncut_data,data_orig)     # Ensure input data not altered by func
        assert (cut_data == data).all()

        # Ensure that passing a nonexistent/misspelled kwarg raises an error
        with pytest.raises(MISSING_ARG_ERRS):
            cut_data = cut_trials(uncut_data, trial_lims, smp_rate=1000, axis=1, foo=None).transpose((2,0,1))

    assert cut_data.shape == data.shape


@pytest.mark.parametrize('data_type',
                         [('spike_timestamp'),('spike_bool')])
def test_realign_data(spike_data, data_type):
    """ Unit tests for realign_data function """
    data, timepts = spike_data[data_type]
    data_orig = data.copy()
    n_trials, n_units = data.shape[:2]

    # Realign timestamps, then realign back to original timebase and test if same
    if data_type == 'spike_timestamp':
        realigned = realign_data(data, 0.5*np.ones((n_trials,)), trial_axis=0)
        realigned = realign_data(realigned, -0.5*np.ones((n_trials,)), trial_axis=0)
        assert object_array_equal(data,data_orig)     # Ensure input data not altered by func
        assert realigned.shape == data.shape
        for trial in range(n_trials):
            for unit in range(n_units):
                assert np.allclose(realigned[trial,unit], data[trial,unit])

        # Test for consistent output with transposed data dimensionality
        realigned = realign_data(data.T, 0.5*np.ones((n_trials,)), trial_axis=-1)
        realigned = realign_data(realigned, -0.5*np.ones((n_trials,)), trial_axis=-1)
        assert object_array_equal(data,data_orig)     # Ensure input data not altered by func
        assert realigned.shape == data.T.shape
        for trial in range(n_trials):
            for unit in range(n_units):
                assert np.allclose(realigned[unit,trial], data[trial,unit])

        # Ensure that passing a nonexistent/misspelled kwarg raises an error
        with pytest.raises(MISSING_ARG_ERRS):
            realigned = realign_data(data, 0.5*np.ones((n_trials,)), trial_axis=0, foo=None)

    # For boolean data, realign to 2 distinct times, then concatenate together and test if same
    else:
        realigned1 = realign_data(data, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                  timepts=timepts, time_axis=-1, trial_axis=0)
        realigned2 = realign_data(data, 0.5*np.ones((n_trials,)), time_range=(0,0.5),
                                  timepts=timepts, time_axis=-1, trial_axis=0)
        realigned = np.concatenate((realigned1,realigned2), axis=-1)
        assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
        assert realigned.shape == data.shape
        assert (realigned == data).all()

        # Test for consistent output with transposed data dimensionality
        realigned1 = realign_data(data.T, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                  timepts=timepts, time_axis=0, trial_axis=-1)
        realigned2 = realign_data(data.T, 0.5*np.ones((n_trials,)), time_range=(0,0.5),
                                  timepts=timepts, time_axis=0, trial_axis=-1)
        realigned = np.concatenate((realigned1,realigned2), axis=0)
        assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
        assert realigned.shape == data.T.shape
        assert (realigned == data.T).all()

        # Ensure that passing a nonexistent/misspelled kwarg raises an error
        with pytest.raises(MISSING_ARG_ERRS):
            realigned1 = realign_data(data.T, 0.5*np.ones((n_trials,)), time_range=(-0.5,-0.001),
                                            timepts=timepts, time_axis=0, trial_axis=-1, foo=None)


@pytest.mark.parametrize('data_type',
                         [('spike_timestamp'),('spike_bool')])
def test_pool_electrode_units(spike_data, data_type):
    """ Unit tests for pool_electrode_units function """

    def _count_all_spikes(data,elecs=None):
        if elecs is None: elecs = np.zeros(data.shape[1])
        elec_set = unsorted_unique(elecs)

        result = np.empty((len(elec_set,)),dtype=int)
        for i_elec,elec in enumerate(elec_set):
            if data_type == 'spike_bool':
                result[i_elec] = data[:,elecs==elec,:].sum()
            else:
                result[i_elec] = np.asarray([len(spikes) for spikes in data[:,elecs==elec].flatten()]).sum()

        return result

    data, timepts = spike_data[data_type]
    result = _count_all_spikes(data)
    # Concatenate data to simulate spiking data from 2 units on 2 electrodes
    data = np.concatenate((data,data), axis=1)
    data_orig = data.copy()
    electrodes = [1,1,2,2]
    n_trials, n_units = data.shape[:2]
    if data_type == 'spike_bool': n_timepts = data.shape[2]
    out_shape = (n_trials,2,n_timepts) if data_type == 'spike_bool' else (n_trials,2)

    data_checker = np.array_equal if data_type == 'bool' else object_array_equal

    # Check if overall spike count, size is consistent after pooling
    data_mua,elec_idxs = pool_electrode_units(data, electrodes, axis=1, return_idxs=True)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert data_mua.shape == out_shape
    assert (elec_idxs == np.array([0,2])).all()
    assert (_count_all_spikes(data,electrodes) == np.array([result,result])).all()

    # Check for consistent output with axis-transposed data
    data_mua,elec_idxs = pool_electrode_units(np.moveaxis(data,1,0), electrodes, axis=0,
                                              return_idxs=True)
    assert data_checker(data,data_orig)     # Ensure input data not altered by func
    assert data_mua.shape == (out_shape[1],out_shape[0],*out_shape[2:])
    assert (elec_idxs == np.array([0,2])).all()
    assert (_count_all_spikes(data,electrodes) == np.array([result,result])).all()


# =============================================================================
# Unit tests for plotting functions
# =============================================================================
def test_plot_mean_waveforms():
    """ Unit tests for plot_mean_waveforms function """
    n_units = 3
    n_spikes = 100
    n_timepts = 1000

    # Use dpss tapers as data to plot, since they are all very distinct looking
    means = compute_tapers(n_timepts, time_width=1.0, freq_width=4, n_tapers=n_units)
    waveforms = np.empty((n_units,), dtype=object)
    for unit in range(n_units):
        waveforms[unit] = means[:,[unit]] + 0.1*np.random.standard_normal((n_timepts,n_spikes))
    waveforms_orig = waveforms.copy()

    # Basic test that plotted data == input data
    lines, _, _ = plot_mean_waveforms(waveforms, plot_sd=True)
    assert object_array_equal(waveforms, waveforms_orig)
    for unit in range(n_units):
        assert np.allclose(lines[unit][0].get_ydata(), waveforms[unit].mean(axis=1))

    # Test w/o SD plot (mean only)
    lines, _, _ = plot_mean_waveforms(waveforms, plot_sd=False)
    assert object_array_equal(waveforms, waveforms_orig)
    for unit in range(n_units):
        assert np.allclose(lines[unit][0].get_ydata(), waveforms[unit].mean(axis=1))

    # Test with data from only one unit
    lines, _, _ = plot_mean_waveforms(waveforms[0], plot_sd=True)
    assert object_array_equal(waveforms, waveforms_orig)
    assert np.allclose(lines[0][0].get_ydata(), waveforms[0].mean(axis=1))

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        lines, _, _ = plot_mean_waveforms(waveforms, foo=None)


def test_plot_waveform_heatmap():
    """ Unit tests for plot_waveform_heatmap function """
    n_units = 1
    n_spikes = 100
    n_timepts = 1000

    # Use dpss tapers as data to plot, since they are all very distinct looking
    means = compute_tapers(n_timepts, time_width=1.0, freq_width=4, n_tapers=n_units)
    waveforms = np.empty((n_units,), dtype=object)
    for unit in range(n_units):
        waveforms[unit] = means[:,[unit]] + 0.1*np.random.standard_normal((n_timepts,n_spikes))
    waveforms_orig = waveforms.copy()

    # Basic test that plotted data == input data
    patch, _ = plot_waveform_heatmap(waveforms)
    assert object_array_equal(waveforms, waveforms_orig)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        patch, _ = plot_waveform_heatmap(waveforms, foo=None)


def test_imports():
    """ Test different import methods for spikes module """
    # Import entire package
    import spynal
    spynal.spikes.density
    # Import module
    import spynal.spikes as spk
    spk.density
    # Import specific function from module
    from spynal.spikes import density
    density
