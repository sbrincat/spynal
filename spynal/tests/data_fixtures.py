"""
Fixtures and functions for generating synthetic data for testing

Overview
--------
Functionality includes:

- pytest data fixtures for use with unit tests
- general-purpose random synthetic data generation functions

Function list
-------------
Fixtures for generating test data of different data schemes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- one_sample_data : Simulate one-sample (single-condition) data
- two_sample_data : Simulate set of two-sample (2 condition) data
- one_way_data : Simulate set of 3-condition data along a single dimension
- two_way_data : Simulate set of 4-condition data along two orthogonal dimensions

Fixtures for generating circular/angular test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- one_sample_circ_data : Simulate one-sample (single-condition) circular data
- paired_circ_data : Simulate paired samples of circular-circular data
- paired_circ_linear_data : Simulate paired samples of circular-linear data
- one_sample_circ_data_parametered : One-sample circ data in deg or rad x circular or axial
- paired_circ_data_parametered : Paired circ-circ data in deg or rad x circular or axial
- paired_circ_linear_data_parametered : Paired circ-linear data in deg or rad x circular or axial

Fixtures for generating spiking test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- spike_timestamp : Simulate set of spike trains as per-trial spike timestamps
- spike_bool : Simulate set of spike trains as per-trial binary trains
- spike_data : "Meta-fixture" that returns both timestamp and boolean spiking data types in a dict
- spike_timestamp_trial_uncut : late set of spike trains as spike timestamps not cut into trial
- spike_bool_trial_uncut : Simulate set of spike trains as binary trains not cut into trials
- spike_data_trial_uncut : "Meta-fixture" that returns both trial-uncut spike data types in a dict
- spike_waveform : Simulate spike waveforms of set of spike trains

Fixtures for generating oscillatory test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- oscillation : Simulate continuous (LFP/EEG-like) oscillatory data
- bursty_oscillation : Simulate bursty continuous oscillatory data
- spiking_oscillation : Simulate spiking oscillatory data
- oscillatory_data : Wrapper dict containing all other oscillation fixture functions

Fixtures for generating synchrony test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- oscillation_pair : Simulate set of field-field pairs of synchronized oscillatory data
- spike_field_pair : Simulate set of spike-field pairs of synchronized oscillatory data

Fixtures for generating multivariate test data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- one_sample_multivariate_data : Simulate one-sample (single-condition) multivariate data
- two_sample_multivariate_data : Simulate set of two-sample (2 condition) multivariate data

Functions for generating synthetic data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- simulate_data : Simulate single-condition random data with given distribution and parameters
- simulate_dataset : Simulates multi-condition random data with effect size, dist'n, parameters

Function reference
------------------
"""
import pytest
from math import pi
import numpy as np

from scipy.stats import norm, poisson, bernoulli
from scipy.stats.mstats import gmean

from spynal.utils import set_random_seed
from spynal.spikes import simulate_spike_trains, simulate_spike_waveforms, times_to_bool
from spynal.spectra import simulate_oscillation
from spynal.sync import simulate_multichannel_oscillation
from spynal.multi import covariance_matrix
from spynal.tests.validity_test_multi import create_subspace, simulate_subspace_dataset

# Possible errors to expect when inputting a missing/misspelled argument
# Used for unit tests against silently ignoring incorrect arguments
MISSING_ARG_ERRS = (TypeError,AttributeError,AssertionError)


# =============================================================================
# Fixtures for generating fixed test data of different data schemes
# =============================================================================
@pytest.fixture(scope='session')
def one_sample_data():
    """
    Fixture simulates one-sample (single-condition) fixed data to use for unit tests

    RETURNS
    data :   ndarray, shape=(10,4) . Simulated data, simulating (10 trials x 4 channels)
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    set_random_seed(1)

    n       = 10
    n_chnls = 4
    mu      = 10.0
    sd      = 5.0
    return norm.ppf(np.random.rand(n,n_chnls), loc=mu, scale=sd)


@pytest.fixture(scope='session')
def two_sample_data():
    """
    Fixture simulates set of two-sample (2 condition) fixed data to use for unit tests

    RETURNS
    data    (20,4) ndarray. Simulated data, simulating (10 trials*2 conditions/levels x 4 channels)
    labels  (20,) ndarray of int8. Set of condition labels corresponding to each trial in data
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    set_random_seed(1)

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
    Fixture simulates set of 3-condition fixed data to use for unit tests of one-way test functions

    RETURNS
    data    (30,4) ndarray. Simulated data, simulating (10 trials*3 conditions x 4 channels)

    labels  (30,) ndarray of int8. Set of condition labels corresponding to each trial in data
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    set_random_seed(1)

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
    Fixture simulates set of 4-condition fixed data to use for unit tests of two-way test functions

    RETURNS
    data    (40,4) ndarray. Simulated data, simulating (10 trials*4 conditions/levels x 4 channels)

    labels  (40,4) ndarray of int8. Set of condition labels corresponding to each trial in data.
            Columns 0,1 correspond to main effects, column 2 corresponds to interaction
    """
    # Note: seed=1 makes data reproducibly match output of Matlab
    set_random_seed(1)

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


# =============================================================================
# Fixtures for generating circular/angular test data
# =============================================================================
@pytest.fixture(scope='session')
def one_sample_circ_data():
    """ Generate set of 1-sample circular data in radians (range [0,2pi]) for unit tests """
    set_random_seed(1) # Note: seed=1 makes data reproducibly match output of Matlab

    n_obs, n_chnls = 50, 10
    data = 2*pi*np.random.rand(n_obs, n_chnls)

    return data


@pytest.fixture(scope='session')
def paired_circ_data():
    """ Generate set of paired-sample circular data in radians (range [0,2pi]) for unit tests """
    set_random_seed(1) # Note: seed=1 makes data reproducibly match output of Matlab

    n_obs, n_chnls = 50, 10
    data1 = 2*pi*np.random.rand(n_obs, n_chnls)
    data2 = 2*pi*np.random.rand(n_obs, n_chnls)

    return data1, data2


@pytest.fixture(scope='session')
def paired_circ_linear_data():
    """ Generate set of paired-sample circular (radians, range [0,2pi]) and linear data """
    set_random_seed(1) # Note: seed=1 makes data reproducibly match output of Matlab

    n_obs, n_chnls = 50, 10
    circ_data = 2*pi*np.random.rand(n_obs, n_chnls) # range [0,2pi]
    linear_data = 100*np.random.rand(n_obs, n_chnls) - 50   # range [-50,+50]

    return circ_data, linear_data


@pytest.fixture(scope='session')
def one_sample_circ_data_parametered(one_sample_circ_data):
    """
    "Meta-fixture" that returns 1-sample circular data in (radians vs degrees) x
    (periodic on full circle vs axially symmetric)

    Returns
    -------
    data : dict {str:ndarray}
        Dictionary mapping data type to data values, for each type of circular data:
        {'radians_circle','degrees_circle','radians_axial','degrees_axial'}

    data2std : dict {str:func}
        Dictionary mapping data type to lambda function used to transform data back to
        standard 'radians_circle' type
    """
    data = one_sample_circ_data
    data = {'radians_circle':   data,
            'degrees_circle':   np.rad2deg(data),
            'radians_axial':    data/2,
            'degrees_axial':    np.rad2deg(data)/2}
    data2std = {'radians_circle':   lambda data: data,
                'degrees_circle':   lambda data: np.deg2rad(data),
                'radians_axial':    lambda data: data*2,
                'degrees_axial':    lambda data: np.deg2rad(data)*2}

    return data, data2std


@pytest.fixture(scope='session')
def paired_circ_data_parametered(paired_circ_data):
    """
    "Meta-fixture" that returns paired-sample circular data in (radians vs degrees) x
    (periodic on full circle vs axially symmetric)

    Returns
    -------
    data1/2 : dict {str:ndarray}
        Dictionaries mapping data type to data values, for each type of circular data:
        {'radians_circle','degrees_circle','radians_axial','degrees_axial'}

    data2std : dict {str:func}
        Single dictionary mapping data type to lambda function used to transform data back to
        standard 'radians_circle' type
    """
    data1, data2 = paired_circ_data

    data1 = {'radians_circle':   data1,
             'degrees_circle':   np.rad2deg(data1),
             'radians_axial':    data1/2,
             'degrees_axial':    np.rad2deg(data1)/2}
    data2 = {'radians_circle':   data2,
             'degrees_circle':   np.rad2deg(data2),
             'radians_axial':    data2/2,
             'degrees_axial':    np.rad2deg(data2)/2}

    data2std = {'radians_circle':   lambda data: data,
                'degrees_circle':   lambda data: np.deg2rad(data),
                'radians_axial':    lambda data: data*2,
                'degrees_axial':    lambda data: np.deg2rad(data)*2}

    return data1, data2, data2std


@pytest.fixture(scope='session')
def paired_circ_linear_data_parametered(paired_circ_linear_data):
    """
    "Meta-fixture" that returns paired-sample circular (rads vs degs) and linear data x
    (periodic on full circle vs axially symmetric)

    Returns
    -------
    circ_data : dict {str:ndarray}
        Dictionaries mapping data type to data values, for each type of data:
        {'radians_circle','degrees_circle','radians_axial','degrees_axial'}
    linear_data : ndarray
        Linear data
    data2std : dict {str:func}
        Single dictionary mapping data type to lambda function used to transform data back to
        standard 'radians_circle' type
    """
    circ_data, linear_data = paired_circ_linear_data

    circ_data = {'radians_circle':   circ_data,
                 'degrees_circle':   np.rad2deg(circ_data),
                 'radians_axial':    circ_data/2,
                 'degrees_axial':    np.rad2deg(circ_data)/2}
    data2std = {'radians_circle':   lambda data: data,
                'degrees_circle':   lambda data: np.deg2rad(data),
                'radians_axial':    lambda data: data*2,
                'degrees_axial':    lambda data: np.deg2rad(data)*2}

    return circ_data, linear_data, data2std


# =============================================================================
# Fixtures for generating spiking test data
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
# Fixtures for generating oscillatory test data
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
    data = (data - data.min()) / np.ptp(data)
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
# Fixtures for generating synchrony test data
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
    return simulate_multichannel_oscillation(2, frequency, amplitude=5.0, phase=[pi/4,0],
                                             phase_sd=[0,pi/4], noise=1.0, n_trials=40,
                                             time_range=1.0, smp_rate=1000, seed=1)


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
    set_random_seed(1)

    # todo code up something actually proper (rate-modulated Poisson process?)
    spkdata,lfpdata = oscillation_pair[:,:,0], oscillation_pair[:,:,1]

    # Convert continuous oscillation to probability (range 0-1)
    spkdata = (spkdata - spkdata.min()) / np.ptp(spkdata)
    spkdata = spkdata**2  # Sparsen high rates some

    # Use probabilities to generate Bernoulli random variable at each time point
    spkdata =  bernoulli.ppf(0.5, spkdata).astype(bool)

    return spkdata, lfpdata


# =============================================================================
# Fixtures for generating multivariate test data
# =============================================================================
@pytest.fixture(scope='session')
def one_sample_multivariate_data():
    """ Generate 1-sample MVN data for unit tests: basis~(3, 2), data~(200, 3), cov~(3, 3) """
    set_random_seed(1) # Note: seed=1 makes data reproducibly match output of Matlab

    basis = create_subspace()
    data,labels = simulate_subspace_dataset(basis=basis)
    data -= data.mean(axis=0, keepdims=True)
    cov = covariance_matrix(data)

    return basis, data, cov, labels

@pytest.fixture(scope='session')
def two_sample_multivariate_data():
    """ Generate 2-sample MVN data for unit tests: basis~(3, 2), data~(200, 3), cov~(3, 3) """
    set_random_seed(1) # Note: seed=1 makes data reproducibly match output of Matlab

    basis = create_subspace()
    data,labels = simulate_subspace_dataset(basis=basis)
    data -= data.mean(axis=0, keepdims=True)
    cov = covariance_matrix(data)

    # Tweak args to create_subspace() to get diff basis for 2nd set of data
    basis2 = create_subspace(angle=30)
    data2,labels = simulate_subspace_dataset(basis=basis2)
    data2 -= data2.mean(axis=0, keepdims=True)
    cov2 = covariance_matrix(data2)

    return basis, data, cov, basis2, data2, cov2, labels


# =============================================================================
# Functions for generating synthetic data
# =============================================================================
def simulate_data(distribution='normal', mean=None, spread=1, n=100, seed=None):
    """
    Simulates random data with given distribution and parameters

    data = simulate_data(distribution='normal', mean=None, spread=1, n=100, seed=None)

    ARGS
    distribution    String. Name of distribution to simulate data from.
                    Options: 'normal' [default] | 'poisson'

    mean            Scalar. Mean/rate parameter for distribution.
                    Default: 0 if distribution='normal' else 10

    spread          Scalar. Spread parameter for distribution.
                    SD for normal; unused for Poisson
                    Default: 1.0

    n               Int | Tuple. Number/shape of random variable draws to simulate.
                    Can set either single integer value or a tuple for shape of RV array.
                    Default: 100

    seed            Int. Random generator seed for repeatable results.
                    Set=None [default] for unseeded random numbers.

    RETURNS
    data            (n,) | (*n) ndarray. Simulated random data.
                    Returns as 1D array if n is an int.
                    Returns with shape given by n if it is a tuple.
    """
    if seed is not None: set_random_seed(seed)

    if mean is None:
        if distribution in ['normal','norm','gaussian','gauss']:    mean = 0.0
        else:                                                       mean = 10

    dist_func = _distribution_name_to_func(distribution)

    return dist_func(n, mean, spread)


def simulate_dataset(gain=5.0, offset=5.0, n_conds=2, n=100, n_chnls=1, distribution='normal',
                     spreads=1.0, correlation=0, seed=None):
    """
    Simulates random data across multiple conditions/groups with given condition effect size,
    distribution and parameters

    data,labels = simulate_dataset(gain=5.0,offset=5.0,n_conds=2,n=100,
                                   distribution='normal',correlation=0,seed=None)

    ARGS
    gain    Scalar | (n_conds,) array-like. Sets the effect size (difference
            in mean value between each condition).
            If scalar, it's the difference in mean between each successive condition.
            If vector, it's the difference in mean from baseline for each individual condition.
            Set = 0 to simulate no expected difference between conditions.
            Default: 5.0

    offset  Scalar. Additive baseline value added to any condition effects. Default: 5.0

    n_conds Int. Number of distinct conditions/groups to simulate. Default: 2

    n       Int. Number of trials/observations to simulate *per condition*. Default: 100

    n_chnls Int. Number of independent data channels to simulate (all channels currently
            have same stochastic properties). Default: 1

    distribution    String. Name of distribution to simulation from.
                    Options: 'normal' [default] | 'poisson'

    spreads Scalar | (n_conds,) array-like. Spread of each condition's data distribution.
            (Gaussian SD; not used for other distributions).
            If scalar, the same spread is used for all conditions.
            If vector, one spread value should be given for each condition.

    correlation Float in range[-1,+1]. Correlation between data in each condition.
            Note: Currently only supported for 2 conditions with normal distribution
            (simulated as multivariate normal w/ covariance matrix based on correlation,spreads)

    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    RETURNS
    data    (n*n_conds,n_chnls). Simulated data for multiple repetitions of one/more conditions.

    labels  (n*n_conds,) of int. Condition/group labels for each trial.
            Sorted in group order to simplify visualization.
    """
    # todo Add ability to simulate independent data series, different n for each cond,
    if seed is not None: set_random_seed(seed)

    # For single-condition data, treat gain as scalar increase over baseline response
    if n_conds == 1:
        gains = np.asarray([gain])
        spreads = np.asarray([spreads])

    else:
        # Single gain = incremental difference btwn cond 0 and 1, 1 and 2, etc.
        # Convert to vector of gains for each condition
        gains = gain*np.arange(n_conds) if np.isscalar(gain) else np.asarray(gain)

        # Convert scalar spread value to vector for each condition
        spreads = spreads*np.ones((n_conds,)) if np.isscalar(spreads) else np.asarray(spreads)

    assert len(gains) == n_conds, \
        ValueError("Vector-valued <gain> must have length == n_conds (%d != %d)"
                   % (len(gains), n_conds))

    assert len(spreads) == n_conds, \
        ValueError("Vector-valued <spreads> must have length == n_conds (%d != %d)"
                   % (len(spreads), n_conds))

    assert (correlation >= -1) and (correlation <= 1), \
        ValueError("Correlation must be in range [-1,+1] (%.2f input)" % correlation)

    if correlation != 0:
        assert (n_conds == 2) and (distribution == 'normal'), \
            ValueError("correlation currently only supported for 2 conds, normal distribution")

    # Final mean value = baseline + condition-specific gain
    means = offset + gains

    # Generate data for each condition and stack together -> (n_trials*n_conds,) array
    if correlation == 0:
        n_ = n if n_chnls == 1 else (n,n_chnls)
        data = np.concatenate([simulate_data(distribution=distribution, mean=mean, spread=spread, n=n_, seed=None)
                               for mean,spread in zip(means,spreads)], axis=0)

    # Generate data for both condition using single multivariate normal distribution
    else:
        # Convert SDs -> variances and compute pooled variance = geometric mean
        variances = np.asarray(spreads)**2
        var_pooled = gmean(variances)
        # Covariance matrix = variances and covariance = pooled variance * correlation
        cov_mx = [[variances[0], var_pooled*correlation], [var_pooled*correlation, variances[1]]]

        # Generate multivariate normal data with given means and covariance matrix
        n_ = (n,) if n_chnls == 1 else (n,n_chnls)
        data = np.random.multivariate_normal(means, cov_mx, size=n_)
        # Reshape (n,n_conds) -> (n*n_conds,)
        data = data.reshape((n*n_conds,), order='F')

    # Create (n_trials_total,) vector of group labels (ints in set 0-n_conds-1)
    labels = np.hstack([i_cond*np.ones((n,)) for i_cond in range(n_conds)])

    return data, labels


# =============================================================================
# Helper functions
# =============================================================================
def _distribution_name_to_func(name):
    """ Converts name of distribution to scipy.stats-based function to generate random variables """
    # Random variables generated in a way that reproducibly matches output of Matlab
    name = name.lower()

    # Normal RV's : mu = mean, s = SD
    if name in ['normal','norm','gaussian','gauss']:
        return lambda n,mu,s: norm.ppf(np.random.rand(n) if np.isscalar(n) else np.random.rand(*n),
                                       loc=mu, scale=s)

    # Poisson RV's : mu = lamba (aka mean,rate), s is unused
    elif name in ['poisson','poiss']:
        return lambda n,mu,s: poisson.ppf(np.random.rand(n) if np.isscalar(n) else np.random.rand(*n),
                                          mu=mu)

    else:
        raise ValueError("%s distribution is not yet supported. Should be 'normal' | 'poisson'")
