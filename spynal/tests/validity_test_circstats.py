"""
Suite of tests to assess "face validity" of circular analysis functions in circstats.py
Usually used to test new or majorly updated functions.

A large battery of these tests are run in the notebook circstats_testing.ipynb
"""
import os
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt

from spynal.tests.data_fixtures import simulate_data, simulate_dataset
from spynal.utils import set_random_seed, iarange
from spynal.circstats import wrap, circ_distance, circ_subtract, \
                             circ_mean, circ_average, \
                             circ_r, circ_rbar, circ_rbar2_unbiased, \
                             circ_var, circ_std, circ_sem, \
                             circ_dispersion, von_mises_kappa, \
                             rayleigh_test, circ_mean_test, circ_ANOVA1, \
                             circ_circ_correlation, circ_linear_correlation, circ_linear_regression
from spynal.helpers import _merge_dicts

# =============================================================================
# Data simulation
# =============================================================================
# HACK Simulate normal data, then wrap into the circle
# todo Should ideally replace these with proper von Mises simulations, I guess
def simulate_angular_data(units='radians', range='circle', **kwargs):
    """
    Simulates 1-sample random angular data with given mean, spread, etc.
    (in radians or degrees, fully circular or axially symmetric)
    """
    data = simulate_data(**kwargs)
    data = wrap(data, degrees=(units == 'degrees'), axial=(range == 'axial'))

    return data


def simulate_paired_angular_data(datatype='circ_circ', units='radians', range='circle',
                                 correlation=0.5, slope=1.0, mean=0, offset=0, **kwargs):
    """
    Simulates correlated pair-sample random angular data with given mean, spread, etc.
    (circular-circular or circular-linear data, in radians or degrees,
    fully circular or axially symmetric)
    """
    mean1 = mean
    mean2 = mean1 + offset

    corr_sign, corr_mag = np.sign(correlation), np.abs(correlation)
    if corr_sign != 0: slope *= corr_sign

    # Randomly set first dataset; set 2nd dataset as weighted average of first + random
    data1 = simulate_data(mean=mean1, **kwargs)
    data2 = slope*(corr_mag*data1 + (1-corr_mag)*simulate_data(mean=mean2, **kwargs))

    # For circular-circular paired data, wrap both dataset into range [0,2pi]
    if datatype in ['circ-circ','circ_circ','circular-circular','circ_circular']:
        data1 = wrap(data1, degrees=(units == 'degrees'), axial=(range == 'axial'))
        data2 = wrap(data2, degrees=(units == 'degrees'), axial=(range == 'axial'))
    # For circular-linear paired data, only wrap data2 (circular data), not data1 (linear data)
    elif datatype in ['circ-linear','circ_linear','circular-linear','circ_linear']:
        data2 = wrap(data2, degrees=(units == 'degrees'), axial=(range == 'axial'))

    return data1, data2


def simulate_angular_dataset(units='radians', range='circle', **kwargs):
    """
    Simulates random data across multiple conditions/groups with given condition effect size,
    distribution and parameters for angular data (in radians or degrees,
    fully circular or axially symmetric)
    """
    data, labels = simulate_dataset(**kwargs)
    data = wrap(data, degrees=(units == 'degrees'), axial=(range == 'axial'))

    return data, labels


# =============================================================================
# Tests
# =============================================================================
def test_circ_diff(stat, test_values=iarange(-90,90,15), do_tests=False):
    """
    Basic testing for functions measuring circular difference:
    :func:`circ_subtract` (signed difference) and :func:`circ_distance` (unsigned diff)

    Generates two sets of data, offset by test_value, and compares measure difference to expected.
    For test failures, raises an error or warning (depending on value of `do_tests`).

    Parameters
    ----------
    stat : str
        Which measure of circular difference to compute: 'subtract'|'distance'

    test_values : array-like, shape=(n_values,), dtype=float, default:iarange(-90,90,15)
        List of signed difference values to test.

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    Returns
    -------
    diff : ndarray, shape=(n_values,n_datapoints)
        Differences btwn datasets for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    theta1 = iarange(-720,720,45)

    func = circ_subtract if stat == 'subtract' else circ_distance

    diff = np.empty((len(test_values),len(theta1)))
    passed = True

    for i_value,test_value in enumerate(test_values):
        theta2 = theta1 - test_value
        diff[i_value,:] = func(theta1, theta2, degrees=True)

        test = np.allclose(diff[i_value,:], test_value) if stat == 'subtract' else \
               np.allclose(diff[i_value,:], abs(test_value))
        if not test:
            message = "Incorrect value(s) found for test = %.1f" % test_value
            AssertionError(message) if do_tests else warn(message)
            passed = False

    return diff, passed


def test_circ_mean(stat='mean', test='mean', test_values=None, n_reps=100, seed=None,
                   do_tests=True, do_plots=False, plot_dir=None, ax=None, **kwargs):
    """
    Basic testing for functions measuring circular central tendency:
    :func:`circ_mean` and :func:`circ_average` (weighted average)

    Generates synthetic data, estimates stat using given method,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    stat : str
        Which basic statistic of central tendency to compute: 'mean'|'average'

    test : str, default: 'mean'
        Type of test to run. Options:

        - 'mean' : Tests multiple values for distribution mean.
            Checks for monotonically increasing mean.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks that mean doesn't vary with spread.
        - 'n': Tests multiple values of number of trials (n).
            Checks that mean doesn't vary with n.

    test_values : array-like, shape=(n_values,), dtype=float
        List of values to test. Interpretation and defaults are test-specific:

        - 'mean' :      Gaussian means. Default: [0,5,15,30,45]
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,5,15,30,45]
        - 'n' :         Trial numbers. Default: [25,50,100,200,400,800]

    n_reps : int, default: 100
        Number of independent repetitions of tests to run

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    **kwargs :
        All other keyword args passed as-is to `simulate_angular_data`

    Returns
    -------
    mean : ndarray, shape=(n_values,)
        Mean estimated stat for each tested value

    sd : ndarray, shape=(n_values,)
        Across-run SD of stat for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)

    test = test.lower()

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(mean=5.0, spread=15.0, n=500, distribution='normal', seed=None,
                    units='degrees', range='circle')
    sim_args = _merge_dicts(sim_args, kwargs)

    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'mean':
        test_values = [0,5,15,30,45] if test_values is None else test_values
        del sim_args['mean']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda mean: simulate_angular_data(**sim_args,mean=mean)

    elif test in ['spread','spreads','sd']:
        test_values = [1,5,15,30,45] if test_values is None else test_values
        del sim_args['spread']                      # Delete preset arg so uses arg to lambda below
        gen_data = lambda spread: simulate_angular_data(**sim_args,spread=spread)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['mean'] = 0     # Set mean=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_angular_data(**sim_args,n=n_trials)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    func = circ_mean if stat == 'mean' else circ_average

    values = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        n = test_value if test in ['n','n_trials','bias'] else sim_args['n']
        args = {'weights':np.random.rand(n)} if stat == 'average' else {}

        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data = gen_data(test_value)
            values[i_value,i_rep] = func(data, degrees=True, **args)

    # Compute mean and std dev across different reps of simulation
    sd = values.std(axis=1,ddof=0)
    mean = values.mean(axis=1)

    if do_plots:
        if ax is None:  plt.figure()
        else:           plt.sca(ax)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, mean, sd, marker='o')
        xlabel = 'n_trials' if test == 'bias' else test
        plt.xlabel(xlabel)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'mean-summary-%s' % test))

    # Determine if test actually produced the expected values
    # 'mean' : Test if stat increases monotonically with between-group mean
    if test == 'mean':
        evals = [((np.diff(mean) >= 0).all(),
                  "Stat doesn't increase monotonically with simulated mean")]

    # 'spread' : Test if stat decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [(np.ptp(mean) < sd.max(),
                  "Stat has larger than expected range across spreads")]

    # 'n' : Test if stat is ~ same for all values of n (unbiased by n)
    elif test in ['n','n_trials']:
        evals = [(np.ptp(mean) < sd.max(),
                  "Stat has larger than expected range across n's (likely biased by n)")]

    else:
        evals = {}

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return mean, sd, passed


def test_circ_spread(stat, method=None, test='mean', test_values=None, n_reps=100, seed=None,
                     do_tests=True, do_plots=False, plot_dir=None, ax=None, **kwargs):
    """
    Basic testing for functions computing various definitions of circular spread (variance, SD, etc.)

    Generates synthetic data, estimates stat using given method,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    stat : str
        Which basic statistic of spread to compute: 'R'|'Rbar'|'var'|'std'|'sem'|'dispersion'|'kappa'

    method : str, default: 'Fisher_Mardia'
        For stat=='var' or 'std', there are multiple options from the stats literature for how to
        define the statistic. Set this to determine which is used (see specific func's for details).

    test : str, default: 'mean'
        Type of test to run. Options:

        - 'mean' : Tests multiple values for distribution mean.
            Checks that value doesn't vary with mean.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks for monotonically increasing mean.
        - 'n': Tests multiple values of number of trials (n).
            Checks that value doesn't vary with n (except for stat='sem')

    test_values : array-like, shape=(n_values,), dtype=float
        List of values to test. Interpretation and defaults are test-specific:

        - 'mean' :      Gaussian means. Default: [0,5,15,30,45]
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,5,15,30,45]
        - 'n' :         Trial numbers. Default: [25,50,100,200,400,800]

    n_reps : int, default: 100
        Number of independent repetitions of tests to run

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    **kwargs :
        All other keyword args passed as-is to `simulate_angular_data`

    Returns
    -------
    mean : ndarray, shape=(n_values,)
        Mean estimated stat for each tested value. For stat='R', 'Rbar', or 'Rbar2', actually
        plots, evaluates and returns 1 - stat, and for stat='kappa', returns 1/stat,
        so sign logic is consistent across all stats.

    sd : ndarray, shape=(n_values,)
        Across-run SD of stat for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)

    test = test.lower()

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(mean=5.0, spread=15.0, n=500, distribution='normal', seed=None,
                    units='degrees', range='circle')
    sim_args = _merge_dicts(sim_args, kwargs)

    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'mean':
        test_values = [0,5,15,30,45] if test_values is None else test_values
        del sim_args['mean']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda mean: simulate_angular_data(**sim_args,mean=mean)

    elif test in ['spread','spreads','sd']:
        test_values = [1,5,15,30,45] if test_values is None else test_values
        del sim_args['spread']                      # Delete preset arg so uses arg to lambda below
        gen_data = lambda spread: simulate_angular_data(**sim_args,spread=spread)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['mean'] = 0     # Set mean=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_angular_data(**sim_args,n=n_trials)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    # Map statistic onto appropriate function to compute it and associated parameters
    stat2func = {'R':circ_r, 'Rbar':circ_rbar, 'Rbar2':circ_rbar2_unbiased,
                 'var':circ_var, 'std':circ_std, 'sem':circ_sem,
                 'dispersion':circ_dispersion, 'kappa':von_mises_kappa}
    func = stat2func[stat]
    args = {'method':method} if stat in ['var','std'] else {}

    values = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data = gen_data(test_value)
            value = func(data, degrees=True, **args)
            if stat == 'kappa': values[i_value,i_rep] = 1/value
            elif 'R' in stat:   values[i_value,i_rep] = 1 - value
            else:               values[i_value,i_rep] = value

    # Compute mean and std dev across different reps of simulation
    sd = values.std(axis=1,ddof=0)
    mean = values.mean(axis=1)

    if do_plots:
        if ax is None:  plt.figure()
        else:           plt.sca(ax)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, mean, sd, marker='o')
        xlabel = 'n_trials' if test == 'bias' else test
        plt.xlabel(xlabel)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'mean-summary-%s' % test))

    # Determine if test actually produced the expected values
    # 'mean' : Test if stat increases monotonically with between-group mean
    if test == 'mean':
        evals = [(np.ptp(mean) < sd.max(),
                  "Stat %s has larger than expected range across mean values" % stat)]

    # 'spread' : Test if stat decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [((np.diff(mean) >= 0).all(),
                  "Stat %s doesn't increase monotonically with simulated spread" % stat)]

    # 'n' : Test if stat is ~ same for all values of n (unbiased by n)
    elif test in ['n','n_trials']:
        if stat in ['R','sem']:
            evals = [((np.diff(mean) <= 0).all(),
                     "Stat %s doesn't decrease monotonically with n" % stat)]
        else:
            evals = [(np.ptp(mean) < sd.max(),
                     "Stat %s has larger than expected range across n's (likely biased by n)" % stat)]

    else:
        evals = {}

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return mean, sd, passed


def test_rayleigh_test(test='spread', test_values=None, stat='p', n_reps=100, seed=None,
                       do_tests=True, do_plots=False, plot_dir=None, ax=None, **kwargs):
    """
    Basic testing for :func:`rayleigh_test` function.

    Generates synthetic data, estimates stat using given method,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    test : str, default: 'mean'
        Type of test to run. Options:

        - 'mean' : Tests multiple values for distribution mean.
            Checks that stat doesn't vary with mean.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks for monotonically increasing stat.
        - 'n': Tests multiple values of number of trials (n).
            Checks for monotonically increasing stat.

    test_values : array-like, shape=(n_values,), dtype=float
        List of values to test. Interpretation and defaults are test-specific:

        - 'mean' :      Gaussian means. Default: [0,5,15,30,45]
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,5,15,30,45]
        - 'n' :         Trial numbers. Default: [25,50,100,200,400,800]

    stat : str, default: 'p'
        Which statistic to evaluate, plot, and return. Options: 'p' | 'Z'
        For `stat`='p', -log(p) is returned, so it's also expected to increase with SNR.

    n_reps : int, default: 100
        Number of independent repetitions of tests to run

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    **kwargs :
        All other keyword args passed as-is to `simulate_angular_data`

    Returns
    -------
    mean : ndarray, shape=(n_values,)
        Mean estimated stat for each tested value. For `stat`=='p', this is -log(p), so
        the sign logic is consistent.

    sd : ndarray, shape=(n_values,)
        Across-run SD of stat for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)

    test = test.lower()

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(mean=5.0, spread=45.0, n=500,
                    distribution='normal', seed=None, units='degrees', range='circle')
    sim_args = _merge_dicts(sim_args, kwargs)

    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'mean':
        test_values = [0,5,15,30,45] if test_values is None else test_values
        del sim_args['mean']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda mean: simulate_angular_data(**sim_args,mean=mean)

    elif test in ['spread','spreads','sd']:
        test_values = [1,5,15,30,45] if test_values is None else test_values
        del sim_args['spread']                      # Delete preset arg so uses arg to lambda below
        gen_data = lambda spread: simulate_angular_data(**sim_args,spread=spread)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['mean'] = 0     # Set mean=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_angular_data(**sim_args,n=n_trials)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    values = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data = gen_data(test_value)
            p, Z = rayleigh_test(data, degrees=True, return_stats=True)
            if stat == 'p':
                values[i_value,i_rep] = -np.log10(p)
            else:
                values[i_value,i_rep] = Z

    # Compute mean and std dev across different reps of simulation
    sd = values.std(axis=1,ddof=0)
    mean = values.mean(axis=1)

    if do_plots:
        if ax is None:  plt.figure()
        else:           plt.sca(ax)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, mean, sd, marker='o')
        xlabel = 'n_trials' if test == 'bias' else test
        plt.xlabel(xlabel)
        plt.ylabel(stat)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'mean-summary-%s' % test))

    # Determine if test actually produced the expected values
    # 'mean' : Test if stat increases monotonically with between-group mean
    if test == 'mean':
        evals = [(np.ptp(mean) < sd.max(),
                  "Stat '%s' has larger than expected range across means" % stat)]

    # 'spread' : Test if stat decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [((np.diff(mean) <= 0).all(),
                  "Stat '%s' doesn't decrease monotonically with spread" % stat)]

    # 'n' : Test if stat increases monotonically with n
    elif test in ['n','n_trials']:
        evals = [((np.diff(mean) >= 0).all(),
                 "Stat '%s' doesn't increase monotonically with n" % stat)]

    else:
        evals = {}

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return mean, sd, passed


def test_circ_mean_test(test='mean', test_values=None, stat='p', n_reps=100, seed=None,
                        do_tests=True, do_plots=False, plot_dir=None, ax=None, **kwargs):
    """
    Basic testing for :func:`circ_mean_test` function.

    Generates synthetic data, estimates stat using given method,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    test : str, default: 'mean'
        Type of test to run. Options:

        - 'mean' : Tests multiple values for distribution mean.
            Checks for monotonically increasing stat.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks for monotonically decreasing stat.
        - 'n': Tests multiple values of number of trials (n).
            Checks that stat doesn't vary with n.
        - 'bias' : Tests multiple n values with 0 mean.
            Checks that stat is not > 0 (unbiased).

    test_values : array-like, shape=(n_values,), dtype=float
        List of values to test. Interpretation and defaults are test-specific:

        - 'mean' :      Gaussian means. Default: [0,5,15,30,45]
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,5,15,30,45]
        - 'n'/'bias' :  Trial numbers. Default: [25,50,100,200,400,800]

    stat : str, default: 'p'
        Which statistic to evaluate, plot, and return. Options: 'p' | 'S'
        For `stat`='p', -log(p) is returned, so it's also expected to increase with SNR.

    n_reps : int, default: 100
        Number of independent repetitions of tests to run

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    **kwargs :
        All other keyword args passed as-is to `simulate_angular_data`

    Returns
    -------
    mean : ndarray, shape=(n_values,)
        Mean estimated stat for each tested value. For `stat`=='p', this is -log(p), so
        the sign logic is consistent.

    sd : ndarray, shape=(n_values,)
        Across-run SD of stat for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)

    test = test.lower()

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(mean=5.0, spread=45.0, n=500,
                    distribution='normal', seed=None, units='degrees', range='circle')
    sim_args = _merge_dicts(sim_args, kwargs)

    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'mean':
        test_values = [0,5,15,30,45] if test_values is None else test_values
        del sim_args['mean']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda mean: simulate_angular_data(**sim_args,mean=mean)

    elif test in ['spread','spreads','sd']:
        test_values = [1,5,15,30,45] if test_values is None else test_values
        del sim_args['spread']                      # Delete preset arg so uses arg to lambda below
        gen_data = lambda spread: simulate_angular_data(**sim_args,spread=spread)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['mean'] = 0     # Set mean=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_angular_data(**sim_args,n=n_trials)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    baseline = 0

    values = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data = gen_data(test_value)
            p, S = circ_mean_test(data, degrees=True, return_stats=True)
            if stat == 'p':
                values[i_value,i_rep] = -np.log10(p)
            else:
                values[i_value,i_rep] = S

    # Compute mean and std dev across different reps of simulation
    sd = values.std(axis=1,ddof=0)
    mean = values.mean(axis=1)

    if do_plots:
        if ax is None:  plt.figure()
        else:           plt.sca(ax)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, mean, sd, marker='o')
        xlabel = 'n_trials' if test == 'bias' else test
        plt.xlabel(xlabel)
        plt.ylabel(stat)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'mean-summary-%s' % test))

    # Determine if test actually produced the expected values
    # 'mean' : Test if stat increases monotonically with between-group mean
    if test == 'mean':
        evals = [((np.diff(mean) >= 0).all(),
                  "Stat '%s' doesn't increase monotonically with mean" % stat)]

    # 'spread' : Test if stat decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [((np.diff(mean) <= 0).all(),
                  "Stat '%s' doesn't decrease monotonically with spread" % stat)]

    # 'n' : Test if stat increases monotonically with n
    elif test in ['n','n_trials']:
        evals = [((np.diff(mean) >= 0).all(),
                  "Stat '%s' doesn't increase monotonically with n" % stat)]

    # 'bias': Test if stat is not > baseline if mean = 0, for varying n
    elif (test == 'bias') and (stat == 'S'):
        evals = [(((mean - baseline) < sd).all(),
                  "Stat '%s' is above baseline for no mean offset" % stat)]

    else:
        evals = {}

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return mean, sd, passed


def test_circ_ANOVA1(test='gain', test_values=None, stat='p', n_reps=100, seed=None,
                     do_tests=True, do_plots=False, plot_dir=None, ax=None, **kwargs):
    """
    Basic testing for :func:`circ_ANOVA1` function.

    Generates synthetic data, estimates stat using given method,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    test : str, default: 'gain'
        Type of test to run. Options:

        - 'gain' : Tests multiple values for between-condition response difference (gain).
            Checks for monotonically increasing stat.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks for monotonically decreasing stat.
        - 'n': Tests multiple values of number of trials (n).
            Checks that stat doesn't vary with n.
        - 'bias' : Tests multiple n values with 0 btwn-cond difference.
            Checks that stat is not > 0 (unbiased).
        - 'n_conds' : Tests multiple values for number of conditions.
            (no actual checking, just to see behavior of info measure)

    test_values : array-like, shape=(n_values,), dtype=float
        List of values to test. Interpretation and defaults are test-specific:

        - 'gain' :      Btwn-condition response differences (gains). Default: [0,5,15,30,45,90]
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,5,15,30,45]
        - 'n'/'bias' :  Trial numbers. Default: [25,50,100,200,400,800]
        - 'n_conds' :   Number of conditions. Default: [2,4,8]

    stat : str, default: 'pev'
        Which statistic to evaluate, plot, and return. Options: 'p' | 'F' | 'pev'.
        For `stat`='p', -log(p) is returned, so it's also expected to increase with SNR.

    n_reps : int, default: 100
        Number of independent repetitions of tests to run

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    **kwargs :
        All other keyword args passed as-is to `simulate_angular_dataset`

    Returns
    -------
    mean : ndarray, shape=(n_values,)
        Mean estimated stat for each tested value. For `stat`=='p', this is -log(p), so
        the sign logic is consistent.

    sd : ndarray, shape=(n_values,)
        Across-run SD of stat for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)

    test = test.lower()

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(gain=15.0, offset=0.0, spreads=90.0, n_conds=4, n=500,
                    distribution='normal', seed=None, units='degrees', range='circle')
    sim_args = _merge_dicts(sim_args, kwargs)

    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'gain':
        test_values = [0,5,15,30,45,90] if test_values is None else test_values
        del sim_args['gain']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda gain: simulate_angular_dataset(**sim_args,gain=gain)

    elif test in ['spread','spreads','sd']:
        test_values = [15,30,45,90] if test_values is None else test_values
        del sim_args['spreads']                     # Delete preset arg so uses arg to lambda below
        gen_data = lambda spreads: simulate_angular_dataset(**sim_args,spreads=spreads)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['gain'] = 0     # Set gain=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_angular_dataset(**sim_args,n=n_trials)

    elif test == 'n_conds':
        test_values = [2,4,8] if test_values is None else test_values
        del sim_args['n_conds']                     # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_conds: simulate_angular_dataset(**sim_args,n_conds=n_conds)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    if stat == 'pev':   baseline = 0
    elif stat == 'p':   baseline = 0.5
    else:               baseline = 1

    values = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        # print(i_value, test_value)

        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data,labels = gen_data(test_value)

            p, stats = circ_ANOVA1(data, labels, degrees=True, return_stats=True, omega=True)
            if stat == 'p':
                values[i_value,i_rep] = -np.log10(p)
            else:
                values[i_value,i_rep] = stats[stat]

    # Compute mean and std dev across different reps of simulation
    sd = values.std(axis=1,ddof=0)
    mean = values.mean(axis=1)

    if do_plots:
        if ax is None:  plt.figure()
        else:           plt.sca(ax)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, mean, sd, marker='o')
        xlabel = 'n_trials' if test == 'bias' else test
        plt.xlabel(xlabel)
        plt.ylabel(stat)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'mean-summary-%s' % test))

    # Determine if test actually produced the expected values
    # 'gain' : Test if stat increases monotonically with between-group gain
    if test == 'gain':
        evals = [((np.diff(mean) >= 0).all(),
                  "Stat '%s' doesn't increase monotonically with btwn-cond mean diff" % stat)]

    # 'spread' : Test if stat decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [((np.diff(mean) <= 0).all(),
                  "Stat '%s' doesn't decrease monotonically with within-cond spread increase" % stat)]

    # 'n' : Test if stat is ~ same for all values of n (unbiased by n)
    elif test in ['n','n_trials']:
        if stat == 'pev':
            evals = [(np.ptp(mean) < sd.max(),
                     "Stat '%s' has larger than expected range across n's (likely biased by n)" % stat)]
        else:
            evals = [((np.diff(mean) >= 0).all(),
                     "Stat '%s' doesn't increase monotonically with n" % stat)]

    # 'bias': Test if stat is not > baseline if gain = 0, for varying n
    elif test == 'bias':
        evals = [(((mean - baseline) < sd).all(),
                  "Stat '%s' is above baseline for no mean difference between conditions" % stat)]

    else:
        evals = {}

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return mean, sd, passed


def test_circ_correlation(corr_type='circ-circ', method='js', stat='rho', test='correlation',
                          test_values=None, n_reps=100, seed=None, do_tests=True,
                          do_plots=False, plot_dir=None, ax=None, **kwargs):
    """
    Basic testing for functions measuring circular correlation: :func:`circ_circ_correlation` and
    :func:`circ_linear_correlation`

    Generates synthetic data, estimates stat using given correlation type and method,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    corr_type : str, default: 'circ-circ'
        Which type of circular correlation to test: 'circ-circ' | 'circ-linear'

    method : str, default: 'js'
        Which method to use to compute circ-circ or circ-linear correlation (see those for details)

    stat : str, default: 'rho'
        Which statistic to test: 'rho' (correlation) | 'p' value

    test : str, default: 'correlation'
        Type of test to run. Options:

        - 'mean' : Tests multiple values for distribution mean.
            Checks that stat doesn't vary with mean.
        - 'offset' : Tests multiple values for difference btwn paired data means.
            Checks that stat doesn't vary with offset.
        - 'correlation' : Tests multiple values for correlation between paired data.
            Checks for monotonically increasing stat.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks that stat doesn't vary with spread.
        - 'n': Tests multiple values of number of trials (n).
            Checks that stat doesn't vary with n.

    test_values : array-like, shape=(n_values,), dtype=float
        List of values to test. Interpretation and defaults are test-specific:

        - 'correlation':Correlation btwn paired data. Default: iarange(-1,1,0.25)
        - 'mean' :      Gaussian means. Default: [0,5,15,30,45]
        - 'offset' :    Difference btwn Gaussian means. Default: iarange(0,360,45)
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,5,15,30,45]
        - 'n' :         Trial numbers. Default: [25,50,100,200,400,800]

    n_reps : int, default: 100
        Number of independent repetitions of tests to run

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    **kwargs :
        All other keyword args passed as-is to `simulate_paired_angular_data`

    Returns
    -------
    mean : ndarray, shape=(n_values,)
        Mean estimated stat for each tested value

    sd : ndarray, shape=(n_values,)
        Across-run SD of stat for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)
    corr_type = corr_type.lower()
    stat = stat.lower()
    test = test.lower()

    if corr_type in ['circ-circ','circ_circ','circular-circular','circ_circular']:
        corr_func = circ_circ_correlation
    elif corr_type in ['circ-linear','circ_linear','circular-linear','circ_linear']:
        corr_func = circ_linear_correlation
    else:
        raise ValueError("Unsupported value '%s' set for <corr_type>" % corr_type)
    corr_args = dict(method=method, degrees=True)
    # if corr_func == circ_circ_correlation: corr_args['method'] = method

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(datatype=corr_type, slope=1.0, correlation=0.5, mean=0.0, spread=30.0, n=500,
                    offset=0.1, distribution='normal', seed=None, units='degrees', range='circle')
    sim_args = _merge_dicts(sim_args, kwargs)

    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'mean':
        test_values = [0,5,15,30,45] if test_values is None else test_values
        del sim_args['mean']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda mean: simulate_paired_angular_data(**sim_args,mean=mean)

    elif test == 'offset':
        test_values = iarange(0,360,45) if test_values is None else test_values
        del sim_args['offset']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda offset: simulate_paired_angular_data(**sim_args,offset=offset)

    elif test in ['spread','spreads','sd']:
        test_values = [1,5,15,30,45] if test_values is None else test_values
        del sim_args['spread']                      # Delete preset arg so uses arg to lambda below
        gen_data = lambda spread: simulate_paired_angular_data(**sim_args,spread=spread)

    elif test == 'correlation':
        test_values = iarange(-1,1,0.25) if test_values is None else test_values
        del sim_args['correlation']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda correlation: simulate_paired_angular_data(**sim_args,correlation=correlation)

    elif test == 'slope':
        test_values = iarange(-1,1,0.25) if test_values is None else test_values
        test_values = test_values[test_values != 0]
        del sim_args['slope']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda slope: simulate_paired_angular_data(**sim_args,slope=slope)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['mean'] = 0     # Set mean=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_paired_angular_data(**sim_args,n=n_trials)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    values = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data1, data2 = gen_data(test_value)
            if stat == 'p':
                _, p = corr_func(data2, data1, return_stats=True, **corr_args)
                values[i_value,i_rep] = -np.log10(p)
            else:
                values[i_value,i_rep] = corr_func(data2, data1,  **corr_args)

    # Compute mean and std dev across different reps of simulation
    sd = values.std(axis=1,ddof=0)
    mean = values.mean(axis=1)

    if do_plots:
        if ax is None:  plt.figure()
        else:           plt.sca(ax)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, mean, sd, marker='o')
        xlabel = 'n_trials' if test == 'bias' else test
        plt.xlabel(xlabel)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'mean-summary-%s' % test))

    # Determine if test actually produced the expected values
    # Test if stat increases monotonically with mean, spread, offset
    if test in ['mean','offset','spread']:
        evals = [(np.ptp(mean) < sd.max(),
                  "Stat %s has larger than expected range across %s values" % (stat,test))]
    # 'n' : Test if stats (other than p value) are ~invariant
    elif test in ['n','n_trials']:
        if stat != 'p':
            evals = [(np.ptp(mean) < sd.max(),
                    "Stat %s has larger than expected range across %s values" % (stat,test))]
        else:
            evals = {}


    # 'correlation' : Test if stat decreases monotonically with within-group spread
    elif test == ['correlation']:
        evals = [((np.diff(mean) >= 0).all(),
                  "Stat %s doesn't increase monotonically with simulated %s" % (stat,test))]
    else:
        evals = {}

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return mean, sd, passed


def test_circ_regression(regress_type='circ-linear', stat='beta', test='correlation', test_values=None,
                         fit_method='gridsearch', error='SSE', n_reps=100, seed=None,
                         do_tests=True, do_plots=False, plot_dir=None, ax=None, **kwargs):
    """
    Basic testing for :func:`circ_linear_regression`

    Generates synthetic data, estimates stat using given regression type,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    regress_type : str, default: 'circ-linear'
        Currently unused. Only 'circ-linear' is supported.

    stat : str, default: 'beta'
        Which statistic to test: 'beta' (fitted coefs) | 'mu' (fitted mean) | 'error'

    test : str, default: 'correlation'
        Type of test to run. Options:

        - 'correlation' : Tests multiple values for correlation between paired data.
            Checks for monotonically decreasing error.
        - 'slope' : Tests multiple values for slope of relationship between paired data.
            Checks for monotonically increasing beta.
        - 'mean' : Tests multiple values for distribution mean.
            Checks that stat doesn't vary with mean.
        - 'offset' : Tests multiple values for difference btwn paired data means.
            Checks that stat doesn't vary with offset.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks that stat doesn't vary with spread.
        - 'n': Tests multiple values of number of trials (n).
            Checks that stat doesn't vary with n.

    test_values : array-like, shape=(n_values,), dtype=float
        List of values to test. Interpretation and defaults are test-specific:

        - 'correlation':Correlation btwn paired data. Default: iarange(0,1,0.1)
        - 'slope' :     Slope of relation btwn paired data. Default: iarange(-5,5)
        - 'mean' :      Gaussian means. Default: [0,5,15,30,45]
        - 'offset' :    Difference btwn Gaussian means. Default: iarange(0,360,45)
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,5,15,30,45]
        - 'n' :         Trial numbers. Default: [25,50,100,200,400,800]

    n_reps : int, default: 100
        Number of independent repetitions of tests to run

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    **kwargs :
        All other keyword args passed as-is to `simulate_paired_angular_data`

    Returns
    -------
    mean : ndarray, shape=(n_values,)
        Mean estimated stat for each tested value

    sd : ndarray, shape=(n_values,)
        Across-run SD of stat for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    assert regress_type == 'circ-linear', "Unsupported values %s set for regress_type" % regress_type

    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)
    regress_type = regress_type.lower()
    stat = stat.lower()
    test = test.lower()

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(datatype=regress_type, correlation=1.0, slope=1.0, mean=0.0, spread=30.0,
                    offset=90.0, n=500, distribution='normal', units='degrees', range='circle',
                    seed=None)
    sim_args = _merge_dicts(sim_args, kwargs)

    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'mean':
        test_values = [0,5,15,30,45] if test_values is None else test_values
        del sim_args['mean']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda mean: simulate_paired_angular_data(**sim_args,mean=mean)

    elif test == 'offset':
        test_values = iarange(0,360,45) if test_values is None else test_values
        del sim_args['offset']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda offset: simulate_paired_angular_data(**sim_args,offset=offset)

    elif test in ['spread','spreads','sd']:
        test_values = [1,5,15,30,45] if test_values is None else test_values
        del sim_args['spread']                      # Delete preset arg so uses arg to lambda below
        gen_data = lambda spread: simulate_paired_angular_data(**sim_args,spread=spread)

    elif test == 'correlation':
        test_values = iarange(0,1,0.1) if test_values is None else test_values
        del sim_args['correlation']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda correlation: simulate_paired_angular_data(**sim_args,correlation=correlation)

    elif test == 'slope':
        test_values = iarange(-5,5) if test_values is None else test_values
        del sim_args['slope']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda slope: simulate_paired_angular_data(**sim_args,slope=slope)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['mean'] = 0     # Set mean=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_paired_angular_data(**sim_args,n=n_trials)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    values = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        # print(test, test_value)
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data1, data2 = gen_data(test_value)
            beta, mu, stats = circ_linear_regression(data2, data1, degrees=True,
                                                     fit_method=fit_method, error=error,
                                                     return_stats=True, return_all_fits=True,
                                                     grid=10 if fit_method == 'gridsearch' else 5)
            if stat == 'beta':  values[i_value,i_rep] = beta
            elif stat == 'mu':  values[i_value,i_rep] = mu
            else:               values[i_value,i_rep] = stats['error']

    # Compute mean and std dev across different reps of simulation
    sd = np.nanstd(values,axis=1,ddof=0)
    mean = np.nanmean(values,axis=1)

    if do_plots:
        if ax is None:  plt.figure()
        else:           plt.sca(ax)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, mean, sd, marker='o')
        xlabel = 'n_trials' if test == 'bias' else test
        plt.xlabel(xlabel)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'mean-summary-%s' % test))

    # Determine if test actually produced the expected values
    # Test if stat increases monotonically with mean, spread, offset
    if test in ['mean','offset','spread','n','n_trials']:
        evals = [(np.ptp(mean) < sd.max(),
                  "Stat %s has larger than expected range across %s values" % (stat,test))]

    # 'correlation' : Test if stat increases monotonically with within-group spread
    elif (test in ['correlation','slope']) and (stat == 'beta'):
        evals = [((np.diff(mean) >= 0).all(),
                 "Stat %s doesn't increase monotonically with simulated %s" % (stat,test))]
    else:
        evals = {}

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        # TEMP elif not cond:  warn(message)

    return mean, sd, passed
