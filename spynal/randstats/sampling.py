# -*- coding: utf-8 -*-
"""
Random sample generators for randomization, permutation (shuffle), and bootstrap statistics

Function reference
------------------
- permutations :                Generate random permutations (resampling w/o replacement)
- bootstraps :                  Generate random bootstrap samples (resampling w/ replacement)
- signs :                       Generate random binary variables (eg for sign tests)
- jackknifes :                  Generate jackknife samples (exclude each observation in turn)
- subsets :                     Generate random length-k subsets (resampling w/o replacement)

Function reference
------------------
"""
import numpy as np

from spynal.utils import set_random_seed


# =============================================================================
# Random sample generators
# =============================================================================
def permutations(n, n_resamples=9999, seed=None):
    """
    Yield generator with a set of `n_resamples` random permutations of integers
    0:n-1, as would be needed for permutation/randomization tests

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int, default: 9999 (appropriate number for test w/ 10,000 samples)
        Number of independent resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Yields
    ------
    resamples : generator, shape=(n_resamples,) of [ndarray, shape=(n,), dtype=int]
        Generator to iterate over for permutation test.
        Each iteration contains a random permutation of integers 0:n-1.
    """
    if seed is not None: set_random_seed(seed)

    for _ in range(n_resamples):
        yield np.random.permutation(n)


def bootstraps(n, n_resamples=9999, seed=None):
    """
    Yield generator with a set of `n_resamples` random resamplings with
    replacement of integers 0:n-1, as would be needed for bootstrap tests or
    confidence intervals

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int, default: 9999 (appropriate number for test w/ 10,000 samples)
        Number of independent resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Yields
    ------
    resamples : generator, shape=(n_resamples,) of [ndarray, shape=(n,), dtype=int]
        Generator to iterate over for boostrap test or confidence interval computation.
        Each iteration contains a random resampling with replacement from integers 0:n-1.
    """
    if seed is not None: set_random_seed(seed)

    for _ in range(n_resamples):
        yield np.random.randint(n, size=(n,))


def signs(n, n_resamples=9999, seed=None):
    """
    Yield generator with a set of `n_resamples` random Bernoulli(p=0.5)
    variables (ie binary 0/1 w/ probability of 0.5), each of length <n>,
    as would be needed to set the signs of stats in a sign test.

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int, default: 9999 (appropriate number for test w/ 10,000 samples)
        Number of independent resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Yields
    ------
    resamples : generator, shape=(n_resamples,) of [ndarray, shape=(n,), dtype=bool]
        Generator to iterate over for random sign test.
        Each iteration contains a random resampling of n Bernoulli random variables.
    """
    if seed is not None: set_random_seed(seed)

    for _ in range(n_resamples):
        yield np.random.binomial(1,0.5, size=(n,)).astype(bool)


def jackknifes(n, n_resamples=None, seed=None):
    """
    Yield generator with a set of n_resamples = n boolean variables,
    each of length n, and each of which excludes one observation/trial in turn,
    as would be needed for a jackknife or leave-one-out test.

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int
        Automatically set=n here. Only included for consistent interface.

    seed : None
        Not used. Only included for consistent interface with other functions.

    Yields
    ------
    resamples : generator, shape=(n,) of [ndarray, shape=(n,), dtype=bool]
        Generator to iterate over for jackknife test.
        Each iteration is all 1's except for a single 0, the observation (trial) excluded
        in that iteration. For the ith resample, the ith trial is excluded.
    """
    assert (n_resamples is None) or (n_resamples == n), \
        ValueError("For jackknife/leave-one-out, n_resamples MUST = n")

    trials = np.arange(n)
    for trial in range(n):
        yield trials != trial


def subsets(n, k, n_resamples=9999, seed=None):
    """
    Yield generator with a set of `n_resamples` random length-k subsets of integers 0:n-1.
    
    Random sampling version of "n-choose-k" function.
    
    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials.

    k : int
        Length of subset to select. For example, k=2 implies random sampling from "n-choose-2".

    n_resamples : int, default: 9999 (appropriate number for test w/ 10,000 samples)
        Number of independent resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Yields
    ------
    resamples : generator, shape=(n_resamples,) of [ndarray, shape=(k,), dtype=int]
        Generator to iterate over for a random-subset test.
        Each iteration contains a random subset of integers 0:n-1.
    """
    if seed is not None: set_random_seed(seed)

    for _ in range(n_resamples):
        yield np.random.permutation(n)[:k]
