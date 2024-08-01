"""
Suite of tests to assess "face validity" of neural information computation functions in info.py
Usually used to test new or majorly updated functions to ensure they perform as expected.

Includes tests that parametrically estimate information as a function of difference in distribution
means, assays of bias, etc. to establish methods produce expected pattern of results.

Plots results and runs assertions that basic expected results are reproduced

Function list
-------------
- test_neural_info :    Contains tests of neural information computation functions
- info_test_battery :   Run standard battery of tests of information computation functions
"""
import os
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt

from patsy import dmatrix

from spynal.tests.data_fixtures import simulate_dataset
from spynal.utils import set_random_seed, condition_mean
from spynal.info import neural_info, neural_info_2groups


def test_neural_info(method, test='gain', test_values=None, distribution='normal',
                     arg_type='label', n_reps=100, seed=None,
                     do_tests=True, do_plots=False, plot_dir=None, **kwargs):
    """
    Basic testing for functions estimating neural information.

    Generates synthetic data, estimates information using given method,
    and compares estimated to expected values.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    method : str
        Name of information function to test: 'pev'|'dprime'|'auroc'|'mutual_information'|'decode'
        Can also test condition_mean function using 'condmean'.
        Can also set to specific 'pev' model type: 'anova1' | 'anova2' | 'regress'

    test : str, default: 'gain'
        Type of test to run. Options:

        - 'gain' : Tests multiple values for between-condition response difference (gain).
            Checks for monotonically increasing information.
        - 'spread' : Tests multiple values for distribution spread (SD).
            Checks for monotonically decreasing information.
        - 'n': Tests multiple values of number of trials (n).
            Checks that information doesn't vary with n.
        - 'bias' : Tests multiple n values with 0 btwn-cond difference.
            Checks that information is not > 0 (unbiased).
        - 'n_conds' : Tests multiple values for number of conditions.
            (no actual checking, just to see behavior of info measure)

    test_values : array-like, shape=(n_values,), dtype=str
        List of values to test. Interpretation and defaults are test-specific:

        - 'gain' :      Btwn-condition response differences (gains). Default: [1,2,5,10,20]
        - 'spread' :    Gaussian SDs for each response distribution. Default: [1,2,5,10,20]
        - 'n'/'bias' :  Trial numbers. Default: [25,50,100,200,400,800]
        - 'n_conds' :   Number of conditions. Default: [2,4,8]

    distribution : str, default: 'normal'
        Name of distribution to simulate data from. Options: 'normal' | 'poisson'

    arg_type : str, default: 'label'
        Which input-argument version of info computing function to use:

        - 'label'   : Standard version with data,labels arguments
        - '2groups' : Binary contrast version with data1,data2 arguments

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
        All other keyword args passed as-is to `simulate_dataset` or information estimation
        function, as appropriate

    Returns
    -------
    info : ndarray, shape=(n_values,)
        Estimated information for each tested value

    sd : ndarray, shape=(n_values,)
        Across-run SD of information for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    # Note: Set random seed once here, not for every random data generation loop below
    if seed is not None: set_random_seed(seed)

    test = test.lower()
    method = method.lower()
    arg_type = arg_type.lower()

    assert arg_type in ['label','labels','2groups'], \
        ValueError("arg_type value %s not supported. Should be 'label' | '2groups'" % arg_type)
    if arg_type == '2groups':
        assert test != 'n_conds', \
            ValueError("Cannot run 'n_conds' test of condition number with '2groups' arg_type")

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    sim_args = dict(gain=5.0, offset=5.0, spreads=5.0, n_conds=2, n=500,
                    distribution=distribution, seed=None)
    # Override defaults with any simulation-related params passed to function
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)

    if test == 'gain':
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['gain']                        # Delete preset arg so uses arg to lambda below
        gen_data = lambda gain: simulate_dataset(**sim_args,gain=gain)

    elif test in ['spread','spreads','sd']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['spreads']                     # Delete preset arg so uses arg to lambda below
        gen_data = lambda spreads: simulate_dataset(**sim_args,spreads=spreads)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['gain'] = 0     # Set gain=0 for bias test
        del sim_args['n']                           # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_trials: simulate_dataset(**sim_args,n=n_trials)

    elif test == 'n_conds':
        test_values = [2,4,8] if test_values is None else test_values
        del sim_args['n_conds']                     # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_conds: simulate_dataset(**sim_args,n_conds=n_conds)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    method_ = method

    # Deal with special-case linear models -- funnel into pev function
    if method in ['pev','regress','anova1','anova2','anovan']:
        method_ = 'pev'
        # For PEV, additional argument to neural_info() specifying linear model to use
        if method == 'pev': kwargs.update({'model':'anova1'})
        else:               kwargs.update({'model':method})

    # For these signed binary methods, reverse default grouping bc in stimulated data,
    # group1 > group0, but signed info assumes opposite preference
    if method in ['dprime','d','cohensd', 'auroc','roc','aucroc','auc']:
        groups = [1,0]
        if (arg_type != '2groups') and ('groups' not in kwargs): kwargs.update({'groups':[1,0]})
    else:
        groups = [0,1]

    # Expected baseline value for no btwn-condition difference = 0.5 for AUROC,
    # 1/n_classes for decode, 0 for other methods
    if method in ['auroc','roc','aucroc','auc']:    baseline = 0.5
    elif method == 'decode':                        baseline = 1/sim_args['n_conds']
    else:                                           baseline = 0

    info = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data,labels = gen_data(test_value)

            # For regression model, convert labels list -> design matrix, append intercept term
            if method == 'regress': labels = dmatrix('1 + C(vbl1,Sum)',{'vbl1':labels})

            # Compute difference btwn condition means
            if method == 'condmean':
                cond_means, _ = condition_mean(data, labels, conditions=[0,1])
                info[i_value,i_rep] = np.diff(cond_means, axis=0)

            # 2-group interface to (some) information methods
            elif arg_type == '2groups':
                info[i_value,i_rep] = neural_info_2groups(data[labels==groups[0]],
                                                          data[labels==groups[1]],
                                                          method=method_, **kwargs)

            # (data, labels) interface to (all) information methods
            else:
                info[i_value,i_rep] = neural_info(data, labels, method=method_, **kwargs)

    # Compute mean and std dev across different reps of simulation
    sd      = info.std(axis=1,ddof=0)
    info    = info.mean(axis=1)

    if do_plots:
        plt.figure()
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, info, sd, marker='o')
        xlabel = 'n' if test == 'bias' else test
        plt.xlabel(xlabel)
        plt.ylabel("Information (%s)" % method_)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'info-summary-%s-%s' % (method,test)))

    # Determine if test actually produced the expected values
    # 'gain' : Test if information increases monotonically with between-group gain
    if test == 'gain':
        evals = [((np.diff(info) >= 0).all(),
                  "Information doesn't increase monotonically with btwn-cond mean diff")]

    # 'spread' : Test if information decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [((np.diff(info) <= 0).all(),
                  "Information doesn't decrease monotonically with within-cond spread increase")]

    # 'n' : Test if information is ~ same for all values of n (unbiased by n)
    elif test in ['n','n_trials']:
        evals = [(info.ptp() < sd.max(),
                  "Information has larger than expected range across n's (likely biased by n)")]

    # 'bias': Test if information is not > baseline if gain = 0, for varying n
    elif test == 'bias':
        evals = [(((info - baseline) < sd).all(),
                  "Information is above baseline for no mean difference between conditions")]

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return info, sd, passed


def info_test_battery(methods=('pev','dprime','auroc','mutual_information','decode','condmean'),
                      tests=('gain','spread','n','bias'), do_tests=True, **kwargs):
    """
    Run a battery of given tests on given neural information computation methods

    Parameters
    ----------
    methods : array-like of str, default: ('pev','dprime','auroc','mutual_information','decode')
        List of neural information methods to test.

    tests : array-like of str
        List of tests to run. Certain combinations of methods,tests are skipped, as they are not
        expected to pass (ie 'n_trials','bias' tests skipped for biased metric 'mutual_info').
                Default: ('gain','spread','n','bias') (all supported tests)

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    **kwargs :
        Any other kwargs passed directly to test_neural_info()
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]

    for test in tests:
        for method in methods:
            print("Running %s test on %s" % (test,method))
            # Skip tests expected to fail due to properties of given info measures
            # (ie ones that are biased/affected by n)
            if (test in ['n','n_trials','bias']) and \
               (method in ['mutual_information','mutual_info']):
                do_tests_ = False
            else:
                do_tests_ = do_tests

            _,_,passed = test_neural_info(method, test=test, do_tests=do_tests_, **kwargs)

            print('%s' % 'PASSED' if passed else 'FAILED')
            # If saving plots to file, let's not leave them all open
            if 'plot_dir' in kwargs: plt.close('all')
