# -*- coding: utf-8 -*-
"""
Multivariate analyses of neural population data -- subspaces/manifolds, information geometry, etc.

Overview
--------
Functionality for multivariate neural analyses, which typically operates at the level of
populations of neural data channels, units, or voxels. Includes analyses for characterizing
and comparing multivariate activity patterns that broadly overlap with what different people
refer to as "subspace", "manifold", or "information/representational geometry" analyses.

Functionality currently includes:
- Useful **linear algebra utilities** not included in numpy/scipy/standard libraries
- Computing different measures of **distance** between population vector representations
- Estimating **dimensionality** of neural population activity
- Computing **alignment/overlap between subspaces** of population activity
- Much more functionality is in progress, and will be added...

Most functions perform operations in a "mass-multivariate" manner. This means
that rather than embedding function calls in for loops over channels, timepoints, etc., like this::

    for condition in conditions:
        for timepoint in timepoints:
            results[timepoint,condition] = compute_something(data[timepoint,condition])

You can instead execute a single call on ALL the data, labeling the axis corresponding to distinct
observations (eg trials, timepoints), and the axis corresponding to distinct multivariate features
(eg neural channels/units/voxels) and it will run in parallel (vectorized) across all other data
array dimensions (eg corresponding to experimental conditions, timepoints, etc.) like this:

``results = compute_something(data, trial_axis, feature_axis)``

Function list
-------------
Linear algebra utilities
^^^^^^^^^^^^^^^^^^^^^^^^
- project_vector : Compute projection of one vector onto another
- orthogonalize_vector : Orthogonalize one vector with respect to another

- orthogonalize_matrix : Generate orthonomal basis from input matrix of basis vectors
- symmetric_orthogonalization : Lowdin symmetric orthogonalization for a matrix of basis vectors
- gram_schmidt_orthogonalization : Gram-Schmidt orthogonalization for matrix of basis vectors

- covariance_matrix : Compute covariance matrix of data matrix or stack of data matrixes
- shrinkage_covariance : Covariance estimator with user-set shrinkage

Multivariate distance-type functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- vector_magnitude : Compute magnitude of one or more data vectors using given metric
- vector_distance : Compute distance btwn one/more pairs of data vectors using given metric
- vector_cosine : Compute cosine of angle between two vectors (or btwn arrays of stacked vectors)

Dimensionality estimation
^^^^^^^^^^^^^^^^^^^^^^^^^
- dimensionality : High-level wrapper function for estimating dimensionality of data
- pc_expvar_dim : Est. dimensionality as number of PCs to reach threshold explained variance
- pc_noise_dim : Est. dimensionality as number of PCs > estimate of noise (cf Machens)
- participation_ratio : Continuous measure of dimensionality based on PCA eigenvalue distribution
- (IN PROGRESS) shatter_dim : Dim ~ number of implementable binary classifications  (cf Rigotti)

Subspace evalutation/manipulation functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- subspace_reconstruction_var : Reconstruction-error-based subspace alignment
- subspace_projection_var : Cross-projection-based subspace alignment
- align_subspaces : Optimally align bases of two subspaces using principal angles method

Subspace overlap/alignment functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- subspace_reconstruction_index : Normalized subspace reconstruction error index (cf. Russo 2020)
- subspace_projection_index : Normalized subspace cross-projection index (cf. Elsayed 2016)
- subspace_error_index : 1 - normalized subspace-to-nullspace cross-projection index (Gokcen 2022)
- subspace_principal_angles : Compute principal angles between subspaces (cf. Gallego 2018)

Cross-validation objects/functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- OddEvenSplit : scikit Splitter-like cv object implementing traditional odd/even trial splits
- BalancedKFold : k-fold cross-validation with trials fully balanced btwn classes in each fold
- DummyCrossValidator : scikit Splitter-like object implementing no-cv (same train/test data)

Function reference
------------------
"""
# Created on Thu Apr 00:05:25 2023
#
# @author: sbrincat
from math import floor, ceil
import numpy as np

from sklearn.covariance import ledoit_wolf, oas
from sklearn.model_selection import BaseCrossValidator

from spynal.utils import set_random_seed, randperm, standardize_array, undo_standardize_array, \
                         standardize_array_3d, undo_standardize_array_3d, \
                         correlation, rank_correlation, condition_mean


# =============================================================================
# Multivariate/Linear algebra utilities
# =============================================================================
def project_vector(v1, v2):
    """
    Compute projection of vector v1 onto vector v2 = proj_v2(v1)

    Parameters
    ----------
    v1 : array-like, shape=(n_elems,)
        Vector to project onto v2

    v2 : array-like, shape=(n_elems,)
        Vector to project v1 onto. Must be same length as v1.

    Returns
    -------
    v_proj : ndarray, shape=(n_elems,)
        Orthogonal projection of v1 onto v2
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    return (np.dot(v1, v2) / np.linalg.norm(v2)**2) * v2


def orthogonalize_vector(v1, v2):
    """
    Orthogonalize vector v1 with respect to vector v2

    Subtract projection of v1 onto v2 from v1:  oproj_v2(v1) = v1 - proj_v2(v1)

    Parameters
    ----------
    v1 : array-like, shape=(n_elems,)
        Vector to orthogonalize with respet to v2

    v2 : array-like, shape=(n_elems,)
        Vector to orthogonalize v1 with respect to. Must be same length as v1.

    Returns
    -------
    v_ortho : ndarray, shape=(n_elems,)
        v1 orthogonalized with respect to v2
    """
    return v1 - project_vector(v1, v2)


def orthogonalize_matrix(X, method='symmetric', rankerr=False, tol=None):
    """
    Orthogonalize a matrix of basis vectors using given method.

    A key difference between methods is symmetric does not prioritize any input vectors
    over others, in that they are affected symmetrically. Other methods prioritize
    earlier (leftmost) vectors in input, in that later vectors are affected more.

    Wrapper around method-specific functions (see those for details).

    Parameters
    ----------
    X : ndarray, shape=(n_elements,n_vectors)
        Set of vectors to orthogonalize (arranged in a matrix, where the vectors are columns).
        For a design matrix, the shape would correspond to (n_observations,n_features).

    method : str, default: 'symmetric'
        Method to use to orthogonalize basis:

        - 'Gram-Schmidt' : Iterative method that retains first vector in input, removes the
            projection of subsequent vectors onto the first, and so on... Thus, it prioritizes
            earlier vectors in input. Uses :func:`.gram_schmidt_orthogonalization`.
        - 'QR' : Decomposes matrix X = QR into orthonormal matrix Q (returned) and upper-triangular
            matrix R (not returned here). Like Gram-Schmidt, it is asymmetric, prioritizing earlier
            columns. For full-rank matrices, QR and Gram-Schmidt are equivalent (but QR is much
            faster, and thus generally preferred). Uses :func:`np.linalg.qr`.
        - 'symmetric' : Non-iterative SVD-based method that finds closest (in squared error)
            orthonormal basis to input basis, has symmetric effects on all basis vectors.
            Uses :func:`.symmetric_orthogonalization`.

    rankerr : bool, default: False
        If True, raises an error if X is not full rank.
        If False, just uses only the first <rank> singular vectors in orthogonalized matrix.

    tol : float, default: (eps*largest eigenvalue)
        Threshold minimal eigenvalue below which eigenvalues are regarded as 0. By default,
        set = max(n_elements,n_vectors)*(largest eigenvalue of X)*(precision for datatype).
        This is the default tolerance used in :func:`np.linalg.matrix_rank`.

    Returns
    -------
    X_ortho : ndarray, shape=(n_elements,n_features) or (n_elements,rank)
        Set of orthonormal basis vectors. `method` = 'symmetric' or 'QR' returns a matrix same
        size as input. Gram-Schmidt returns matrix with shape[1] reduced to rank of input matrix
        (implicitly, the rest of the rank-n_features columns would be all 0's).
    """
    method = method.lower()
    if method in ['symmetric','lowdin']:
        return symmetric_orthogonalization(X, rankerr=rankerr, tol=tol)

    elif method == 'qr':
        return np.linalg.qr(X)[0]

    elif method in ['qr','gram-schmidt','gramschmidt','gs']:
        return gram_schmidt_orthogonalization(X, rankerr=rankerr, tol=tol)

    else:
        raise ValueError("Unsupported value '%s' set for `method`" % method)


def gram_schmidt_orthogonalization(X, rankerr=False, tol=None):
    """
    Orthogonalize a matrix of basis vectors using Gram-Schmidt orthogonalization.

    Takes as input a linearly independent set of vectors (the columns of X) and
    outputs an orthonormal set.

    Note: This method (unlike :func:`.symmetric_orthogonalization`) prioritizes earlier
    vectors (columns from left to right) in input basis X. The first output vector is
    = first input vector. The rest are computed by removing their projections onto all
    previous vectors.

    Parameters
    ----------
    X : ndarray, shape=(n_elements,n_vectors)
        Set of vectors to orthogonalize (arranged in a matrix, where the vectors are columns).
        For a design matrix, the shape would correspond to (n_observations,n_features).

    rankerr : bool, default: False
        If True, raises an error if X is not full rank.
        If False, just uses only the first <rank> singular vectors in orthogonalized matrix.

    tol : float, default: (eps*largest eigenvalue)
        Threshold minimal eigenvalue below which eigenvalues are regarded as 0. By default,
        set = max(n_elements,n_vectors)*(largest eigenvalue of X)*(precision for datatype).
        This is the default tolerance used in :func:`np.linalg.matrix_rank`.

    Returns
    -------
    X_ortho : ndarray, shape=(n_elements,rank)
        Set of orthonormal basis vectors.
    """
    n_elements,n_features = X.shape
    rank = np.linalg.matrix_rank(X, tol=tol)
    is_full_rank = rank >= np.min(X.shape)
    if not is_full_rank and rankerr:
        raise ValueError("Input matrix `X` has rank %d (< maxrank=%d). Fix or set rankerr=False."
                         % (rank,n_features))

    X = X / np.linalg.norm(X, axis=0)   # Ensure input basis is normalized

    X_ortho = np.empty((n_elements,rank))

    # Use Gram-Schmidt to find rest of basis vectors
    for j in range(rank):
        # Initialize current output vector = input vector (for 1st vector, it remains=this)
        X_ortho[:,j] = X[:,j]

        # Subtract projections of current vector onto already determined orthonormal basis
        # Note: for j=0, this loop is skipped and output remains = input
        for k in range(j):
            X_ortho[:,j] -= project_vector(X[:,j], X_ortho[:,k])

    # Normalize output basis
    X_ortho = X_ortho / np.linalg.norm(X_ortho, axis=0)

    return X_ortho


def symmetric_orthogonalization(X, rankerr=False, tol=None):
    """
    Orthogonalize a matrix of basis vectors using Lowdin symmetric orthogonalization.

    Like traditional orthogonalization methods (eg Gram-Schmidt/QR), this takes as input a
    linearly independent set of vectors (the columns of X) and outputs an orthonormal set.

    Unlike traditional methods, this doesn't prioritize any vectors in set -- all are
    symmetrically affected -- and the result is as close as possible to the original data
    matrix (in terms of Frobenius norm of difference).

    The Lowdin orthogonalization of X is simply U*V.T, for U*S*V.T = svd(X).

    Parameters
    ----------
    X : ndarray, shape=(n_elements,n_vectors)
        Set of vectors to orthogonalize (arranged in a matrix, where the vectors are columns).
        For a design matrix, the shape would correspond to (n_observations,n_features).

    rankerr : bool, default: False
        If True, raises an error if X is not full rank.
        If False, just uses only the first <rank> singular vectors in orthogonalized matrix.

    tol : float, default: (eps*largest eigenvalue)
        Threshold minimal eigenvalue below which eigenvalues are regarded as 0. By default,
        set = max(n_elements,n_vectors)*(largest eigenvalue of X)*(precision for datatype).
        This is the default tolerance used in :func:`np.linalg.matrix_rank`.

    Returns
    -------
    X_ortho : ndarray, shape=(n_elements,n_vectors)
        Set of orthonormal basis vectors.

    References
    ----------
    https://booksite.elsevier.com/9780444594365/downloads/16755_10030.pdf
    """
    # Compute SVD of data matrix
    U,S,V = np.linalg.svd(X, full_matrices=False)

    # Check that data matrix has full rank
    if tol is None:
        tol = max(X.shape)*S[0]*(np.finfo(X.dtype).eps) # Tolerance level
    rank = np.sum(S > tol) # Number of eigenvalues larger than tolerance
    is_full_rank = rank >= np.min(X.shape)
    if not is_full_rank:
        # If set, raise error if rank < full rank
        if rankerr:
            raise ValueError("Data matrix is not full rank (rank=%d). Can't compute orthogonalization"
                             % rank)
        # If not set, only use first <rank> singular vectors in orthogonalized data matrix
        else:
            U = U[:,:rank]
            V = V[:rank,:]

    return np.dot(U,V)


lowdin_orthogonalization = symmetric_orthogonalization
""" Alias of :func:`.symmetric_orthogonalization`. See there for details. """

def covariance_matrix(data, axis=-2, feature_axis=-1, method='unbiased', **kwargs):
    """
    Compute feature covariance matrix of data matrix or stack of data matrixes.

    Provides unified interface to a number of different methods/functions for computing
    covariance, including the standard empirical estimate (:func:`np.cov`), Ledoit-Wolf shrinkage
    (:func:`sklearn.covariance.ledoit_wolf`), and Oracle Approximating Shrinkage
    (:func:`sklearn.covariance.oas`).

    Shrinkage-based estimators (eg Ledoit-Wolf, OAS) can provide better (lower variance, but
    slightly biased) estimates of covariance when data is ill-conditioned by interpolating
    between the empirical measured covariance and a simpler form (eg diagonal matrix;
    see References).

    Parameters
    ----------
    data : ndarray, shape=(n_obs,n_features) or (...,n_obs,...,n_features,...)
        Data matrix(es) to compute covariance matrix of.

        Can be given as either a single data matrix of shape=(n_obs,n_features) (eg where
        observations might correspond to trials or time points, and features might correspond
        to neural channels/units) or a stack of such data matrices (eg for different conditions).
        In the latter case, the observation and feature axes must be specified in `axis` and
        `feature_axis`, respectively, and the covariance will be computed separately for each
        data matrix, and returned in a single array.

    axis : int, default: -2 (2nd to last axis)
        Axis of data corresponding to distinct observations (eg trials, timepoints, conditions)
        to compute covariance *across*.

    feature_axis : int, default: -2 (2nd to last axis)
        Axis of data corresponding to distinct features (eg neural channels/units) to compute
        covariance *between*.

    method : string or callable, default: 'unbiased'
        Method to use to compute covariance:

        - 'unbiased' : Empirical covariance unbiased by n, using :func:`np.cov` with bias=False
        - 'MLE' : Maximum Likelihood Est of cov (biased by n), using :func:`np.cov` with bias=True
        - 'shrinkage' : Performs "shrinkage" (regularization) of covariance toward
            simpler-structured version, with user-set level of shrinkage/regularization, using
            :func:`shrinkage_covariance`
        - 'ledoit_wolf' : Ledoit-Wolf method, which finds optimal shrinkage that minimizes
            Mean Squared Error btwn estimated and actual covariance,
            using :func:`sklearn.covariance.ledoit_wolf`
        - 'OAS' : Oracle Approximating Shrinkage. Slightly improved regularization method.
            Uses :func:`sklearn.covariance.oas`.

    **kwargs
        Any additional arguments passed as-is to covariance computing function

    Returns
    -------
    cov : ndarray, shape=(n_features,n_features) or (...,n_features,...,n_features,...)
        Feature covariance matrix(es) of data.

        For single data matrix, returns single cov matrix of shape (n_features,n_features).

        For stack of data matrices, returns cov of all data matrices in a single array with
        same shape as input `data`, but with observation `axis` now having length=n_features.

    References
    ----------
    https://scikit-learn.org/stable/modules/covariance.html
    """
    # Convert symbolic string method to callable function/lambda
    if not callable(method):
        method = method.lower()
        # Unbiased standard covariance estimation using np.cov
        if method in ['cov','np.cov','unbiased']:
            cov_func = lambda X: np.cov(X, rowvar=False, bias=False, **kwargs)
        # Biased (maximum likelihood) estimatation using np.cov
        elif method in ['mle','biased']:
            cov_func = lambda X: np.cov(X, rowvar=False, bias=True, **kwargs)
        # Shrinkage estimator with hand-set shrinkage target and parameter
        elif method == 'shrinkage':
            cov_func = lambda X: shrinkage_covariance(X, **kwargs)
        # Ledoit-Wolf shrinkage method (from sklearn.covariance)
        elif method in ['ledoit_wolf','ledoit-wolf']:
            cov_func = lambda X: ledoit_wolf(X, **kwargs)[0]
        # Oracle approximating shrinkage method (from sklearn.covariance)
        elif method in ['oas']:
            cov_func = lambda X: oas(X, **kwargs)[0]
        else:
            raise ValueError('Unsupported option ''%s'' given for covariance <method>' % method)

    if data.ndim < 3:
        if (axis not in [0,-2]) or (feature_axis not in [1,-1]): data = data.T
        cov = cov_func(data)

    else:
        data,shape,ndim = standardize_array_3d(data, axis1=axis, axis2=feature_axis,
                                               target_axis1=-2, target_axis2=-1)
        [n_series,n_obs,n_features] = data.shape

        cov = np.empty((n_series,n_features,n_features))
        # Compute covariance of each stacked data matrix
        for i_series in range(n_series):
            cov[i_series,:,:] = cov_func(data[i_series,:,:])

        cov = undo_standardize_array_3d(cov, shape, ndim, axis1=axis,
                                        axis2=feature_axis, target_axis1=-2, target_axis2=-1)

    return cov


def shrinkage_covariance(data, target='diagonal', param=0.1):
    """
    Estimate feature covariance matrix of data matrix or stack of data matrixes using
    shrinkage estimator with user-set shrinkage value.

    Shrinkage-based estimators can provide better estimates of covariance when data is
    ill-conditioned by interpolating between the empirical measured covariance and a simpler
    form (eg diagonal matrix). To set optimal shrinkage given your data matrix instead of
    hand-set shrinkage, use :func:`sklearn.covariance.ledoit_wolf` or
    :func:`sklearn.covariance.oas`.

    Note: This function is not set up to compute covariance in a "mass-multivariate" fashion.
    For that, use the wrapper function :func:`covariance_matrix` with `method` = 'shrinkage.

    Parameters
    ----------
    data : ndarray, shape=(n_obs,n_features)
        Data matrix to compute feature covariance matrix of.

    target : {'diagonal','scalar','identity'}, default: 'diagonal'
        Target simpler form of covariance matrix to interpolate with empirical
        covariance estimate. Options:

        - 'diagonal' : Matrix w/ sample variances on diagonal, zeros everywhere else
        - 'scalar' : Identity matrix scaled by average sample variance
        - 'identity' : Identity matrix (1's on diagonal, 0's everywhere else)

    param : float, default: 0.1
        Shrinkage parameter determining how much of target covariance is combined with
        empirical covariance estimate. Range is 0-1 (0 is no shrinking, returns empirical
        covariance; 1 is maximal shrinkage, returns covariance target).

    Returns
    -------
    cov : ndarray, shape=(n_features,n_features)
        Feature covariance matrix of data, with given shrinkage
    """
    # Compute empirical covariance
    cov_empirical = np.cov(data, rowvar=False, bias=True)
    n_features = cov_empirical.shape[0]

    # Set up covariance target to shrink toward
    if target == 'diagonal':
        cov_target = np.diag(np.diag(cov_empirical))
    elif target in ['scalar','variance']:
        cov_target = (np.trace(cov_empirical)/n_features) * np.eye(n_features)
    elif target == 'identity':
        cov_target = np.eye(n_features)
    else:
        raise ValueError('Unsupported option ''%s'' given for covariance <target>' % target)

    # Combine empirical covariance and shrinkage target
    return (1-param) * cov_empirical + param * cov_target


# =============================================================================
# Low-level multivariate distance-type functions
# =============================================================================
def vector_magnitude(data, axis=-1, method='euclidean', data_test=None, cov_inv=None,
                     keepdims=True):
    """
    Compute vector magnitude (length/distance from origin) of one or more vectors in
    `data` array, using given distance metric.

    Can optionally cross-validated (or cross-condition) magnitude metric using both
    `data` and `data_test`. Unlike uncrossed metrics, these are unbiased (have expected
    value of 0).

    Parameters
    ----------
    data : ndarray, shape=(n_features,) or (...,n_features,...)
        Data array to compute magnitude of. Should be single data vector (eg activity across
        a set of neural channels/units), or a set of stacked vectors for multiple data series
        (eg timepoints, conditions, etc.). Shape is arbitrary, but `axis` must correspond to
        array dimension in which vector elements (features) are laid out along (used to compute
        vector magnitude), while all other dimensions are treated as separate vectors and
        analyzed independently.

    axis : int, default:-1 (last dimension of data)
        Axis of data array to perform analysis on, in which each vector's elements (features)
        are laid out along. Usually corresponds to distinct neural data channels/units.
        Analysis is performed independently along all other array axes.

    method : str, default: 'euclidean'
        Distance metric to compute magnitudes under. For cross-validated metrics (if a value for
        `data_test` is given), only squared metrics {'sqeuclidean','sqmahalanobis'} allowed
        (otherwise you'd have complex results). Options:

        - 'euclidean' :  Euclidean distance (aka L2-norm)
        - 'sqeuclidean' : Squared Euclidean distance
        - 'mahalanobis :  Mahalanobis distance from origin -- distance relative to covariance
            (like a multivariate z-score). Must input `cov_inv`.
        - 'sqmahalanobis' : Squared Mahalanobis distance. Must input `cov_inv`.

    data_test : ndarray, shape=(n_features,) or(...,n_features,...)
        To compute cross-validated (or cross-condition) magnitude btwn distinct "training" and
        "testing" data, input a distinct data array here. Shape must be same as `data`.

    cov_inv : ndarray, shape=(...,n_features,n_features,...)
        Inverse covariance matrix(es) corresponding to vectors in `data`. Required for method =
        '[sq]mahalanobis', but ignored otherwise. Shape must be same as `data`, but with
        additional axis of length `n_features` inserted immediately after `axis` (together,
        these two axes correspond to the covariance matrixes).

    keepdims : bool, default: True
        If True, axis `axis` is reduced to length-one axes, but retained in output.
        If False, axis `axis` is removed from output.

    Returns
    -------
    magnitude : float or ndarray, shape=(...,[1,]...)
        Magnitude of each vector in `data` (or crossed magnitude of data & data_test).

        For a single vector, returned as a scalar float. For set of multiple vectors,
        returned as array same size as `data`, but with `axis` reduced to length 1
        (if `keepdims` is True) or removed (if `keepdims` is False).
    """
    assert method in ['euclidean','euclid','sqeuclidean','sqeuclid',
                      'mahalanobis','mahal','sqmahalanobis','sqmahal'], \
        ValueError("Unsupported value '%s' set for `method`. "
                   "Must be in Euclidean or Mahalanobis family." % method)

    # Simply call distance function with 1 vector/vector-set = 0
    return vector_distance(data, 0, axis=axis, method=method, cov_inv=cov_inv, keepdims=keepdims,
                           data_test=data_test, data2_test=0 if data_test is not None else 0)


def vector_distance(data, data2, axis=-1, method='euclidean', data_test=None, data2_test=None,
                    cov_inv=None, keepdims=True):
    """
    Compute distance between of one or more pairs of vectors in `data` and `data2` arrays,
    using given distance metric.

    Can optionally cross-validated (or cross-condition) distance metric using both
    `data` and `data_test`. Unlike uncrossed metrics, these are unbiased (have expected
    value of 0).

    Parameters
    ----------
    data,data2 : ndarray, shape=(n_features,) or (...,n_features,...)
        Paired data arrays to compute distance between. Should each be a single data vector
        (eg activity across a set of neural channels/units), or a set of stacked vectors for
        multiple data series (eg timepoints, conditions, etc.).

        Shape is arbitrary, but `axis` must correspond to array dimension in which vector elements
        (features) are laid out along (used to compute distance), while all other dimensions are
        treated as separate vectors and analyzed independently.

    axis : int, default:-1 (last dimension of data)
        Axis of data array to perform analysis on, in which each vector's elements (features)
        are laid out along. Usually corresponds to distinct neural data channels/units.
        Analysis is performed independently along all other array axes.

    method : str, default: 'euclidean'
        Distance metric to compute distances under. For cross-validated metrics (if a value for
        `data_test` is given), only squared metrics {'sqeuclidean','sqmahalanobis'} allowed
        (otherwise you'd have complex results). Options:

        - 'euclidean' :  Euclidean distance (aka L2-norm)
        - 'sqeuclidean' : Squared Euclidean distance
        - 'mahalanobis :  Mahalanobis distance from origin -- distance relative to covariance
            (like a multivariate z-score). Must input `cov_inv`.
        - 'sqmahalanobis' : Squared Mahalanobis distance. Must input `cov_inv`.
        - 'cosine' : Cosine of angle between feaure vectors
        - 'correlation' : Pearson correlation between feature vectors
        - 'spearman' : Spearman rank correlation between feature vectors

    data_test,data2_test : ndarray, shape=(n_features,) or (...,n_features,...)
        To compute cross-validated (or cross-condition) magnitude btwn distinct "training" and
        "testing" data, input a distinct data array here. Shape must be same as `data`.

    cov_inv : ndarray, shape=(...,n_features,n_features,...)
        Inverse covariance matrix(es) corresponding to vectors in `data`. Required for method =
        '[sq]mahalanobis', but ignored otherwise. Shape must be same as `data`, but with
        additional axis of length `n_features` inserted immediately after `axis` (together,
        these two axes correspond to the covariance matrixes).

    keepdims : bool, default: True
        If True, axis `axis` is reduced to length-one axes, but retained in output.
        If False, axis `axis` is removed from output.

    Returns
    -------
    distance : float or ndarray, shape=(...,[1,]...)
        Distance between each vector pair in `data` vs `data2` (or crossed distance
        between data/data_test vs data_test/data2_test).

        For a single pair of vectors, returned as a scalar float. For set of multiple vector pairs,
        returned as array same size as `data`, but with `axis` reduced to length 1
        (if `keepdims` is True) or removed (if `keepdims` is False).
    """
    data = np.asarray(data)
    data2 = np.asarray(data2)
    method = method.lower()

    assert np.isscalar(data2) or np.all(data.shape == data2.shape), \
        "data and data2 must have same shape"

    # Are we computing cross-validated distance across separate training and testing data?
    crossed = data_test is not None
    if crossed:
        data_test = np.asarray(data_test)
        data2_test = np.asarray(data2_test)
        assert np.all(data.shape == data_test.shape), "data and data_test must have same shape"
        assert np.isscalar(data2_test) or np.all(data_test.shape == data2_test.shape), \
            "data_test and data2_test must have same shape"

    # Reshape data array -> (n_data_series,n_features) matrix
    if data.ndim > 1:
        data, data_shape = standardize_array(data, axis=axis, target_axis=-1)
        if crossed: data_test,_ = standardize_array(data_test, axis=axis, target_axis=-1)
        if 'mahalanobis' in method:
            cov_inv,_,_ = standardize_array_3d(cov_inv, axis1=axis, axis2=axis+1,
                                               target_axis1=-2, target_axis2=-1)

    # Non-cross-validated distance metrics
    if not crossed:
        if method in ['euclidean','euclid']:
            distance = np.linalg.norm(data - data2, axis=-1, keepdims=keepdims)

        elif method in ['sqeuclidean','sqeuclid']:
            distance = (data - data2) @ (data_test - data2_test)

        elif method in ['mahalanobis','mahal']:
            distance = np.sqrt((data - data2) @ cov_inv @ (data - data2))

        elif method in ['sqmahalanobis','sqmahal']:
            distance = (data - data2) @ cov_inv @ (data - data2)

        elif method in ['cosine','cosin']:
            distance = vector_cosine(data, data2, axis=axis, keepdims=keepdims)

        elif method in ['correlation','corr','pearson','pearsonr']:
            distance = correlation(data, data2, axis=axis, keepdims=keepdims)

        elif method in ['spearman','rankcorrelation','rankcorr']:
            distance = rank_correlation(data, data2, axis=axis, keepdims=keepdims)

        else:
            raise ValueError("Unsupported value '%s' set for `method`") % method

    # Cross-validated distance metrics
    else:
        if method == 'sqeuclidean':
            distance = (data - data2) @ (data_test - data2_test)

        elif method == 'sqmahalanobis':
            distance = (data - data2) @ cov_inv @ (data_test - data2_test)

        else:
            raise ValueError("Unsupported value '%s' set for `method`. \
                             Only squared distances alowed for cross-validate distance metrics"
                             % method)

    if data.ndim > 1:
        distance = undo_standardize_array(distance, data_shape, axis=axis, target_axis=-1)

    return distance


def vector_cosine(x1, x2, axis=0, keepdims=True):
    """
    Compute cosine of angle between two vectors.
    Can also compute mass-multivariate cosine between two paired sets of stacked vectors.

    To compute angle itself do: np.arccos(vector_cosine(x1,x2)) (in radians),
    or in degrees, np.rad2deg(np.arccos(vector_cosine(x1,x2)))

    Parameters
    ----------
    x1,x2 : array-like, shape=(n_elems,) or (...,n_elems,...)
        Two vectors to compute cosine of angle between, or two array of stacked vectors
        Can be any arbitrary length, but must both be same along `axis`.

    axis : int, default: 0 (1st axis)
        For stacked-vector arrays, this gives the array axis to treat as separate vectors,
        and compute cosine along. Not used for single vector inputs.

    keepdims : bool, default: True
        For stacked-vector arrays, if True, the length-1 reduced vector `axis` is retained
        in the output; if False, `axis` is removed.

    Returns
    -------
    cosine : float or ndarray, shape=(...[,1],...)
        Cosine of angle between vectors. Ranges from [-1,+1].
        Returned as a single float for vector inputs. For stacked-vector arrays, returned
        as an array with same shape as inputs, but with `axis` reduced to length 1 (keepdims=True)
        or removed (keepdims=False).
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    if x1.ndim == 1:
        # Normalize vectors (L2 vector norm)
        x1_u = x1 / np.linalg.norm(x1)
        x2_u = x2 / np.linalg.norm(x2)

        # Compute dot product of norm'd vectors = cosine of angle btwn them
        return np.dot(x1_u, x2_u)

    else:
        # Normalize vectors (L2 vector norm)
        x1_u = x1 / np.linalg.norm(x1, axis=axis, keepdims=True)
        x2_u = x2 / np.linalg.norm(x2, axis=axis, keepdims=True)

        # Compute dot product of norm'd vectors = cosine of angle btwn them
        return (x1_u*x2_u).sum(axis=axis, keepdims=keepdims)


# =============================================================================
# Dimensionality estimation functions
# =============================================================================
def dimensionality(data, labels=None, method='pc_noise', **kwargs):
    """
    High-level interface for computing dimensionality of data with respect to a given
    set of task/behavioral conditions/variables

    NOTE: Unlike most other Spynal functions, this is currently only set up to accept and
    compute dimensionality for a single data matrix (ie, it doesn't yet compute in a
    "mass-multivariate" fashion -- TODO!!!)

    Parameters
    ----------
    data : ndarray, shape=(n_obs,n_features)
        Data to estimate dimensionality of. "Observations" usually correspond to trials
        (and/or timepoints), and "features" usually corresponds to different neural channels/units.

    labels : array-like, shape=(n_obs,), default: None
        Condition labels for each observation (trial) in data. Must be discrete-valued.

        Some dimensionality methods require a `labels` argument, and measure a supervised
        "coding dimensionality" (ie of the data condition means). For others, this argument
        is optional, and if it is omitted, they measured an unsupervised overall data
        dimensionality (ie of the raw data, including noise). See `method` and specific
        functions for details.

    method : str, default: 'pc_noise'
        Method to use to compute dimensionality. Estimate dimensionality as:

        - 'pc_expvar' : Number of PCA eigenvalues > given threshold explained variance,
            using :func:`.pc_expvar_dim`
        - 'pc_noise' : The number of data "signal" PCA eigenvalues > estimated noise
            eigenvalues, cf. Machens 2010, using :func:`.pc_noise_dim`
        - 'participation_ratio' : Continuous measure of dimensionality based on distribution
            of PCA eigenvalues, cf. Gao 2017, using :func:`.participation_ratio`
        - 'shatter' : The number of implementable binary classifications of data
            (aka "shatter dimension") cf. Rigotti 2013, using :func:`.shatter_dim`
            NOTE: This functionality is currently in progress, and not working just yet (TODO)

    **kwargs
        Any other keyword args passed directly to specific function for computing dimensionality

    Returns
    -------
    dimensionality : float
        Estimate of data dimensionality

    References
    ----------
    - Machens, Romo, Brody (2010) J Neurosci https://doi.org/10.1523/JNEUROSCI.3276-09.2010
    - Rigotti...Miller,Fusi (2013) Nature https://doi.org/10.1038/nature12160
    - Gao...Shenoy, Ganguli (2017) https://doi.org/10.1101/214262
    """
    method = method.lower()
    if method in ['pc_noise','pcnoise','machens']:
        dim_func = pc_noise_dim
    elif method in ['pc_expvar','pcexpvar']:
        dim_func = pc_expvar_dim
    elif method in ['participation_ratio','participationratio','pr']:
        dim_func = participation_ratio
    elif method in ['shatter','rigotti']:
        raise ValueError("Rigotti/Fusi shatter dimensionality will be available soon...")
        # dim_func = shatter_dim
    else:
        raise ValueError("Unsupported value '%s' input for `method`" % method)

    return dim_func(data, labels, **kwargs)


def participation_ratio(data, labels=None, **kwargs):
    """
    Estimate dimensionality of data using the "participation ratio" of the eigenspectrum of
    the covariance matrix, from Gao 2017. This is the ratio of the square of the sum of all
    eigenvalues, divided by the sum of the squared eigenvalues::

        dim = sum(lambda)**2 / sum(lambda**2)

    where lambda's are the eigenvalues of the covariance matrix, and the sum is over all of them

    This results in a measure of dimensionality that ranges continuously from 1 (all variance
    concentrated in a single eigenvalue, ie 1 dimension) to the max number of dimensions =
    min(n_observations,n_features) (all eigenvalues equal).

    This can be used to estimate either the unsupervised overall data dimensionality (ie of the
    raw data, including noise; if `labels` is None) or the supervised "coding dimensionality"
    (of the condition means; if values given for `labels`).

    Parameters
    ----------
    data : ndarray, shape=(n_obs,n_features)
        Data to estimate dimensionality of. "Observations" usually correspond to trials
        (and/or timepoints), and "features" usually corresponds to different neural channels/units.

    labels (optional) : array-like, shape=(n_obs,), default: None (raw data dimensionality)
        Condition labels for each observation (trial) in data. Must be discrete-valued.

        If a value us input for `labels`, function will compute mean of data within each condition
        (unique label value) before computing dimensionality, and thus returning an estimate
        of the "coding dimensionality".

        Leave `labels` = None to compute the dimensionality of the raw data (or if input `data`
        already reflects condition means).

    **kwargs :
        Any additional keywords arguments passed to :func:`.covariance_matrix`, which is used to
        compute data covariance (and defaults to :func:`np.cov`)

    Returns
    -------
    dimensionality : float
        Estimate of data dimensionality, ranging continuously from 1 - min(n_obs,n_features)

    References
    ----------
    Gao...Shenoy, Ganguli (2017) https://doi.org/10.1101/214262
    """
    # Compute data condition means, if `labels` input
    if labels is not None:
        data,_ = condition_mean(data, labels)

    # Compute data covariance matrix ~ X.T*X ~ (n_features,n_features)
    cov = covariance_matrix(data, **kwargs)
    # Compute SVD of covariance matrix
    _, S, _ = np.linalg.svd(cov)

    # Participation ratio index
    return np.sum(S)**2 / np.sum(S**2)


def pc_expvar_dim(data, labels=None, cutoff=0.95, **kwargs):
    """
    Estimate dimensionality of data as the first k principal components to retain a given
    cumulative level of explained variance. This results in a discrete measure of dimensionality
    ranging from 1 to the max number of data dimensions = min(n_observations,n_features).

    This can be used to estimate either the unsupervised overall data dimensionality (ie of the
    raw data, including noise; if `labels` is None) or the supervised "coding dimensionality"
    (of the condition means; if values given for `labels`).

    Parameters
    ----------
    data : ndarray, shape=(n_obs,n_features)
        Data to estimate dimensionality of. "Observations" usually correspond to trials
        (and/or timepoints), and "features" usually corresponds to different neural channels/units.

    labels (optional) : array-like, shape=(n_obs,), default: None (raw data dimensionality)
        Condition labels for each observation (trial) in data. Must be discrete-valued.

        If a value us input for `labels`, function will compute mean of data within each condition
        (unique label value) before computing dimensionality, and thus returning an estimate
        of the "coding dimensionality".

        Leave `labels` = None to compute the dimensionality of the raw data (or if input `data`
        already reflects condition means).

    cutoff : float, default: 0.95
        Threshold proportion of explained variance to use to compute dimensionality.
        Dimensionality is the minimum number of PCs needed to explain >= `cutoff` variance.

    **kwargs :
        Any additional keywords arguments passed to :func:`.covariance_matrix`, which is used to
        compute data covariance (and defaults to :func:`np.cov`)

    Returns
    -------
    dimensionality : float
        Estimate of data dimensionality, ranging discretely from 1 - min(n_obs,n_features)

    References
    ----------
    Cunningham & Yu (2014) Nat Neurosci https://doi.org/10.1038/nn.3776
    """
    # Compute data condition means, if `labels` input
    if labels is not None:
        data,_ = condition_mean(data, labels)

    # Compute data covariance matrix ~ X.T*X ~ (n_features,n_features)
    cov = covariance_matrix(data, **kwargs)
    # cov = np.cov(data.T)
    # Compute SVD of covariance matrix and their cumulative proportion of explained variance
    _, S, _ = np.linalg.svd(cov)
    cum_exp_var_ratio = np.cumsum(S) / np.sum(S)

    return np.nonzero(cum_exp_var_ratio >= cutoff)[0][0] + 1


def pc_noise_dim(data, labels, noise_method='sd', cutoff=None, which_eigs='all',
                 n_noise_resmps=100):
    """
    Estimate dimensionality of data as the number of eigenvalues (as from a Principal Components
    Analysis) in full data which can be attributed to data "signal", as opposed to "noise",
    which is estimated by resampling random pairs of within-condition trials, cf. Machens 2010.

    This method is only applicable to the supervised "coding dimensionality"; it is not defined
    for the full dimensionality of raw data. It returns a discrete measure of dimensionality
    ranging from 1 to the max number of data dimensions = min(n_observations,n_features)-1.

    Parameters
    ----------
    data : ndarray, shape=(n_obs,n_features)
        Data to estimate dimensionality of. "Observations" usually correspond to trials
        (and/or timepoints), and "features" usually corresponds to different neural channels/units.

    labels : array-like, shape=(n_obs,)
        Condition labels for each trial in data. Must be discrete-valued, and (unlike some
        dimensionality methods) must be input

    noise_method : {'expvar','quantile','sd'}, default: 'sd'
        Determines specific method to use to compare signal to noise eigenvalues,
        in order to estimate dimensionality:

        - 'expvar' : Find first signal eigenvalue where > `cutoff` proportion of total *signal*
            variance (sum of all signal eigenvalues) is explained. This is the original version
            used in Machens 2010. Our testing with synthetic data suggests it underestimates
            when dimensionality is high relative to number of conditions.
        - 'quantile' : Find first signal eigenvalue that drops below `cutoff` quantile of
            resampled noise eigenvalue distribution. For this and 'sd', comparisons can be done
            against the noise distribution of only the *first* eigenvalue or each signal eigenvalue
            can be compared against the corresponding-order noise eigenvalue, depending on value of
            `which_eigs`. This version (with `which_eigs` = 'first) was used in Supplemental results
            of Rigotti 2013.
        - 'sd' : Find first signal eigenvalue that drops below mean + `cutoff` std dev's
            of resampled noise eigenvalue distribution. In our simulations, 'quantile' and 'sd'
            are less biased than the 'expvar' method, and 'sd'/`which_eigs` = 'all' is slightly less
            biased and lower variance (which is why it's the default).

    cutoff : float, default: (depends on `noise_method`)
        Criterion value to use for determining dimensionality.
        Interpretation and default value depends on value of `noise_method`:

        - 'expvar' : Proportion of signal variance that eigenvalues must cumulatively explain
            (range: 0-1, default: 0.95)
        - 'quantile' : Quantile of noise eigenvalue distribution that signal must drop below
             (range: 0-1, default: 0.95)
        - 'sd' : Multiple of SD of noise eigenvalue dist'n that signal must drop below
             (range: 0-inf, default: 2)

    which_eigs : {'first','all'}, default: 'all'
        Which noise eigenvalue(s) to compare observed signal eigenvalues to,
        to establish noise floor for dimensionality computation:

        - 'first' : Compare each signal eigenvalue to distn of first (largest) noise eigenvalue
        - 'all' : Compare each signal eigenvalue to distn of corresponding-rank noise eigenvalue.

        Note that this parameter is only used for `noise_method` = 'quantile' or 'sd'; for 'expvar',
        all eigenvalues are always used.

    n_noise_resmps : int, default: 1000
        Number of times to randomly resample noise to generate distribution of noise eigenvalues

    Returns
    -------
    dimensionality : int
        Estimate of data dimensionality, ranging discretely from 1 - min(n_obs,n_features)

    References
    ----------
    - Machens, Romo, Brody (2010) J Neurosci https://doi.org/10.1523/JNEUROSCI.3276-09.2010
    - Rigotti...Miller,Fusi (2013) Nature https://doi.org/10.1038/nature12160

    Notes
    -----
    This function is based on Matlab code graciously provided by Mattia Rigotti. Go cite his paper^
    """
    noise_method = noise_method.lower()
    which_eigs = which_eigs.lower()
    assert noise_method in ['expvar','quantile','sd'], \
        "Unsupported value '%s' input for `noise_method`" % noise_method
    assert which_eigs in ['first','all'], \
        "Unsupported value '%s' input for `which_eigs`" % which_eigs

    conditions = np.unique(labels)
    n_conditions = len(conditions)
    assert n_conditions != len(labels), \
        "Number of conditions cannot = number of trials \
            (likely condition-mean data or continuous-valued labels were input"

    if cutoff is None:
        cutoff = 0.95 if noise_method in ['expvar','quantile'] else 2

    # Max number of >0 eigenvalues = smaller of #observations (conditions) and #features (channels)
    n_obs,n_features = data.shape
    n_eigs = min(n_conditions,n_features) - 1

    n_eigs_noise = n_eigs if (which_eigs == 'all') or (noise_method == 'expvar') else 1

    # Condition-mean activity ~ (n_conds,n_features)
    cond_means,_ = condition_mean(data, labels)

    # Total (signal+noise) covariance matrix -> (n_features,n_features) (eg (n_chnls,n_chnls))
    cov = covariance_matrix(cond_means, method='biased')

    # Signal eigenvalues (explained variance) via SVD -> (n_eigs,)
    # Note: eigenvalues are returned in ascending order, so must reverse and select desired top k
    # todo Speed test np.linalg.eigvalsh vs scipy.linalg.eigvalsh (which can return subset)
    total_eigs = np.linalg.eigvalsh(cov)[-1:-(n_eigs+1):-1]

    # Find all trials corresponding to each condition in `labels` and their count
    cond_trials = [np.nonzero(labels == cond)[0] for cond in conditions]
    n_cond_trials = [len(trial_list) for trial_list in cond_trials]
    # Normalization constant for each condition = sqrt(2*n)
    cond_norm = np.sqrt(n_cond_trials)

    ## Resample noise eigenvalues several times using Machens 2010 method
    # Generate random resamplings of noise matrix `eta`
    eta = np.empty((n_conditions,n_features))
    # Eigenvalues of covariance of noise matrix composed of randomly sampled difference
    # in activity btwn trials within same condition, normalized by sqrt(2*n_condition_trials)
    noise_eigs = np.empty((n_eigs_noise,n_noise_resmps))

    for i_resmp in range(n_noise_resmps):
        for i_cond,cond in enumerate(conditions):
            # Randomly select 2 observations/trials from current condition
            smp_trials = cond_trials[i_cond][randperm(n_cond_trials[i_cond], 2)]

            # Compute difference btwn within-cond trials, normalized by sqrt(2*n)
            # to get estimate of noise in cond means
            d = data[smp_trials[0],:] - data[smp_trials[1],:]
            eta[i_cond,:] = d / cond_norm[i_cond]

        # Compute estimated noise covariance
        cov = covariance_matrix(eta, method='biased')
        # Compute estimated noise eigenvalues (explained variance) via SVD -> (n_eigs,n_resmps)
        # Note: eigenvals are returned in ascending order, so must reverse and select desired top k
        noise_eigs[:,i_resmp] = np.linalg.eigvalsh(cov)[-1:-(n_eigs_noise+1):-1]

    # Estimate of dim = # of signal eigenvalues needed to explain given % of signal variance
    if noise_method == 'expvar':
        # Noise floor estimate = mean of resampled noise eigenvalues
        noise_floor = noise_eigs.mean(axis=1)
        # Estimate of signal eigenvalues = total eigs - noise eigs
        signal_eigs = total_eigs - noise_floor

        # Comparison threshold is given proportion of signal variance (sum of signal eigenvalues)
        threshold = cutoff*np.sum(signal_eigs)
        comparison = np.cumsum(signal_eigs)
        # Find signal eigenvalues where given proportion of total signal variance is explained
        idx = np.nonzero(comparison > threshold)[0]

    # Estimate of dim = # of signal eigenvalues > estimated noise floor
    else:
        # Noise floor estimate = given pctile of distribution of given noise eigenvalues
        if noise_method == 'quantile':
            noise_floor = np.quantile(noise_eigs, cutoff, axis=1)
        # Noise floor estimate = given z-score (# of SDs) of distribution of given noise eigenvalues
        else:
            noise_floor = noise_eigs.mean(axis=1) + cutoff*noise_eigs.std(axis=1)

        # Find eigenvalues that drop below noise floor threshold
        idx = np.nonzero(total_eigs < noise_floor)[0]

    # If NO eigenvalues meet cutoff criterion, then dimensionality must be maximal for data
    if len(idx) == 0:
        dim = n_eigs
    else:
        dim = idx[0] + 1 if noise_method == 'expvar' else idx[0]

    return dim


# =============================================================================
# Subspace evaluation and manipulation functions
# =============================================================================
def subspace_reconstruction_var(data, basis, feature_axis=-1, sum_axis=None, expansion_basis=None,
                                keepdims=True):
    """
    Compute the proportion of total variance in data captured by a subspace (given by its
    basis/eigenvector matrix), quantified as the sum of squared error of the subspace basis's
    reconstruction of the data, normalized by the sum of squares of the data itself::

        expvar = 1 - ||X - X * W * W.T||^2 / ||X||^2

    where X is the original data, W is the subspace basis, and ||.|| is the L2 norm

    Note that the basis could be estimated from the data itself, or from a different set of data
    (eg different timepoint, experimental condition), in which case this would reflect the
    cross-variance captured.

    This is used in the "subspace overlap" index of Russo 2020, which computes the cross-variance
    captured in one "reference" set of data by a basis fit to another "comparison set of data,
    normalized by the "self-variance" captured in the reference data by a basis fit to itself.
    This index is implemented in :func:`subspace_reconstruction_index`.

    Function allows mass-multivariate computation of captured variance for multiple data vectors
    (eg different trials/conditions, time points, etc.), but only for a single subspace basis.

    Parameters
    ----------
    data : ndarray, shape=(n_features,) or (...,n_features,...)
        Data to compute captured variance of. Can be single data vector (typically corresponding
        to a population of neural channels/units) or stack of such vectors in an array (eg for
        different trials, timepoints, etc.), with the array axis corresponding to different vector
        features (eg channels) given in `feature_axis`. In this case, variance captured is
        computed separately for each data vector and returned in a similiarly-shaped array.

    basis : ndarray, shape=(n_features,n_components)
        Basis matrix for subspace, which is used to reduce data dimensionality down to
        `n_components`. Column vectors correspond to sequence of orthonormal basis vectors for
        subspace, eg the top-k eigenvectors from a PCA; features along each column correspond to
        same feature in `data` (eg neural channels/units). For simplicity, only a single basis
        is allowed as input.

    feature_axis : int, default: -1 (last axis)
        Axis of `data` corresponding to different features, to compute
        captured variance along. Typically corresponds to different neural channels/units.

    sum_axis : int, default: None (no other axis)
        Optional *additional* axis to compute captured variance along, in addition to
        `feature_axis` (ie using Frobenius matrix norm along two dims instead of standard
        L2/Euclidean norm along only `feature_axis`).

        For index used in Russo 2020, this would correspond to the time axis of the data array;
        for trial-based data, this might correspond to its trial axis.

    expansion_basis : ndarray, shape=(n_components,n_features), default: basis.T
        Basis matrix for re-expansion of dimensionality-reduced data back to original data
        dimensionality. Only needed for dimensionality reduction methods (eg dPCA) that estimate
        distinct bases for dimensionality reduction and re-expansion (the "decoder" and "encoder"
        axes, respectively, of Kobak et al 2016).

        Otherwise, this defaults to the transpose of the subspace `basis` (as is the case for
        PCA, LDA, and most standard dimensionality reduction methods).

    keepdims : bool, default: False
        If True, retains reduced `feature_axis` (and `sum_axis`) as length-one axis(s) in output.
        If False, removes reduced `feature_axis` (and `sum_axis`) from outputs.

    Returns
    -------
    expvar : float or ndarray, shape=(...,[1,]...)
        Proportion of variance in data captured by subspace, based on its reconstruction of the
        original data. Ranges from 0 (no variance captured) to 1 (all variance captured).

        Returns a float for single vector input. For stacked vectors, returns array the same
        shape as `data` but with `feature_axis` (and `sum_axis` if set) reduced to length 1
        (if `keepdims` is True) or removed (if `keepdims` is False).

    References
    ----------
    - Gallego ... Miller (2018) Nature Comms https://doi.org/10.1038/s41467-018-06560-z
    - Russo ... M.Churchland (2020) Neuron https://doi.org/10.1016/j.neuron.2020.05.020
    - Kobak ... Machens (2016) eLife https://doi.org/10.7554/eLife.10989 (dPCA)
    """
    if expansion_basis is None: expansion_basis = basis.T
    if feature_axis < 0: feature_axis = data.ndim + feature_axis
    if (sum_axis is not None) and (sum_axis < 0): sum_axis = data.ndim + sum_axis

    # Move feature axis to end of data array (and optional sum axis just before that)
    if data.ndim > 1:
        if sum_axis is not None:
            data,shape,ndim = standardize_array_3d(data, axis1=sum_axis, axis2=feature_axis,
                                                   target_axis1=-2, target_axis2=-1, reshape=False)
        else:
            data,shape = standardize_array(data, axis=feature_axis, target_axis=-1)

    # Sum of squared errors of reconstruction error from basis / Sum of squares of data
    # Best reconstruction of data from basis
    # ~ ([...,]n_features) * (n_features,n_components) * (n_components,n_features) -> ([...,]n_features)
    data_hat = data @ basis @ expansion_basis

    # Sum of squared errors of reconstruction error from basis
    reduce_axes = -1 if sum_axis is None else (-2,-1)
    ss_error = np.sum((data - data_hat)**2, axis=reduce_axes) # -> (...,)
    # Total Sum of squares of data -> (...,)
    ss_data = np.sum(data**2, axis=reduce_axes)

    # Proportion of data variance captured = 1 - fractional error variance
    expvar = 1 - (ss_error / ss_data)

    # Move data axis back to original location in data array
    if data.ndim > 1:
        # Append singleton axis(s) to replace reduced feature_axis (and sum_axis)
        if sum_axis is None:    expvar = expvar[...,np.newaxis]
        else:                   expvar = expvar[...,np.newaxis,np.newaxis]

    if data.ndim > 1:
        if sum_axis is not None:
            expvar = undo_standardize_array_3d(expvar, shape, ndim,
                                               axis1=sum_axis, axis2=feature_axis,
                                               target_axis1=-2, target_axis2=-1, reshape=False)
        else:
            expvar = undo_standardize_array(expvar, shape, axis=feature_axis, target_axis=-1)

        if not keepdims:
            if sum_axis is None:    expvar = expvar.squeeze(axis=feature_axis)
            else:                   expvar = expvar.squeeze(axis=(feature_axis,sum_axis))
        if isinstance(expvar,np.ndarray) and (expvar.size == 1): expvar = expvar.item()
    else:
        expvar = expvar.squeeze() # Squeeze length-1 array -> float

    return expvar


def subspace_projection_var(cov, basis, feature_axis=(-2,-1),
                            keep_components=False, normalization='pev', keepdims=True):
    """
    Compute the proportion of total variance in data (given by its covariance matrix)
    captured by a subspace (given by its basis/eigenvector matrix), by computing the
    (trace of) the projection of the covariance matrix onto the basis::

        expvar = trace(W.T * C * W) / norm

    where C is the data covariance matrix, W is the subspace basis, and norm is the normalizaiton
    constant (which is determined by `normalization`); if keep_components is True, trace() is
    replaced by diag()

    Note that the basis could be estimated from the data itself, or from a different set of data
    (eg different timepoint, experimental condition), in which case this would reflect the
    cross-variance captured.

    Output can be returned as raw captured data variance (in units of data^2), normalized by
    the number of features (ie channels/neurons, in units of data^2 per feature, cf. Murray 2017),
    or normalized by total data variance (so reflects proportion of variance captured, range 0-1),
    depending on value set for `normalization`. Can also compute variance captured *by each basis
    vector* (rather than total for full basis) using `keep_components` = True.

    This measure is taken from Murray 2017, who used the 'feature' normalization version
    (expressing captured variance per neuron). It's also used to compute the numerator (and in our
    formulation, also the denominator) of the "subspace alignment" index of Elsayed 2016 (cf.
    :func:`.subspace_projection_index`).

    Function allows mass-multivariate computation of captured variance for multiple data vectors
    (eg different trials/conditions, time points, etc.), but only for a single subspace basis.

    Parameters
    ----------
    cov : ndarray, shape=(n_features,n_features) or (...,n_features,n_features,...)
        Cross-feature covariance matrix of data to compute captured variance of.

        Can be single covariance matrix (typically corresponding to covariance of a population of
        neural channels/units) or stack of such matrixes (eg for different trials, timepoints,
        etc.), with the pair of array axes corresponding to different vector features
        (eg channels/neurons) given in `feature_axis`.

    basis : ndarray, shape=(n_features,n_components)
        Basis matrix for subspace, which is used to reduce data dimensionality down to
        `n_components`. Column vectors correspond to sequence of orthonormal basis vectors for
        subspace, eg the top-k eigenvectors from a PCA; features along each column correspond to
        same feature in `data` (eg neural channels/units). For simplicity, only a single basis
        is allowed as input.

    feature_axis : array-like, shape=(2,), default: (-2,-1) (last two array axes)
        Axes of `cov` corresponding to different data vector features. Must have 2 values.
        This will typically correspond to covariance between different neural channels/neurons.

    keep_components : bool, default: False
        If True, returns variance captured by each basis component vector (eg PC) separately.
        If False, return total (summed) variance captured by all basis vectors.

    normalization : {'none','feature','pev'}, default: 'pev'
        How to normalize returned captured variance:

        - 'none' : Return raw, unormalized captured variance (eg in (spikes/s)^2)
        - 'feature' : Normalize by number of vector features (ie channels/neurons),
            so reflects variance per channel/neuron (eg in (spikes/s)^2 per neuron)
        - 'pev' :  Normalize by total variance, so reflects proportion of total data
            variance captured (range 0-1)

    keepdims : bool, default: True
        If True, retains reduced `feature_axis` as length-one axes in output.
        If False, removes reduced `feature_axis` from outputs.

    Returns
    -------
    expvar : float or ndarray, shape=(...,[1,1,],...) or (...,[n_components,n_components,],...)
        Data variance captured by given basis, either raw variance or proportion, depending on
        value set for `normalization`. If `keep_components` is True, returns variance captured by
        each basis component vector; if False, returns variance summed across all components.


        If `keep_components` is False, returns a float for single matrix input. For stacked
        matrices, returns array the same shape as `data` but with `feature_axis` reduced to
        length 1 (if `keepdims` is True) or removed (if `keepdims` is False). If `keep_components`
        is True, returns array the same shape as `data`.

    References
    ----------
    - Murray ... Wang (2017) PNAS https://doi.org/10.1073/pnas.1619449114
    - Elsayed ... Cunningham (2016) Nature Comms https://doi.org/10.1038/ncomms13239
    - Yoo & Hayden (2020) Neuron https://doi.org/10.1016/j.neuron.2019.11.013

    todo Should we retain covariance,basis args here or make it data,basis and compute cov internally?
         Would align args with reconstruction func and simplify the 2-axis feature_axis arg here
    """
    assert len(feature_axis) == 2, "Two axes must be input for `feature_axis` of covariance matrix"
    normalization = normalization.lower()
    n_features, n_components = basis.shape

    # Stack of covariance matrixes -- shift feature axes to end of array
    if cov.ndim > 2:
        cov,shape,ndim = standardize_array_3d(cov, axis1=feature_axis[0], axis2=feature_axis[1],
                                              target_axis1=-2, target_axis2=-1)

    # Compute variance in data captured by subspace as diag/trace(B.T * C * B)
    # ~ (n_components,n_features)*([...,]n_features,n_features)*(n_features,n_components)
    # -> ([...,]n_components,n_components)
    # -> For stacked matrixes, append singleton axis(es) to expand dimensionality to ([...,]1,1)
    # Variance captured by each basis vector = diag(n_components,n_components) -> (n_components,)
    if keep_components:
        expvar = np.diagonal(basis.T @ cov @ basis, axis1=-2, axis2=-1)
        if cov.ndim > 2: expvar = expvar[...,np.newaxis]
    # *Total* variance captured = trace(n_components,n_components) -> float
    else:
        expvar = np.trace(basis.T @ cov @ basis, axis1=-2, axis2=-1)
        if cov.ndim > 2: expvar = expvar[...,np.newaxis,np.newaxis]

    # Return expvar normalized by total variance (ie as proportion of variance captured)
    if normalization == 'pev':
        if keep_components:
            norm = np.diagonal(cov, axis1=-2, axis2=-1)
            norm = norm[...,:n_components] # Need to reduce norm to length matching n_components
            if cov.ndim > 2: norm = norm[...,np.newaxis]
        else:
            norm = np.trace(cov, axis1=-2, axis2=-1)
            if cov.ndim > 2: norm = norm[...,np.newaxis,np.newaxis]

        expvar = expvar / norm

    # Return expvar normalized by number of vector features (neural channels/units)
    elif normalization in ['feature','channel','neuron']:
        expvar = expvar / basis.shape[0]

    else:
        assert normalization in ['none','raw'], \
            ValueError("Unsupported value '%s' set for `normalization`" % normalization)


    # If stacked cov matrixes input, reshape output to original data shape
    if cov.ndim > 2:
        expvar = undo_standardize_array_3d(expvar, shape, ndim,
                                           axis1=feature_axis[0], axis2=feature_axis[1],
                                           target_axis1=-2, target_axis2=-1)
        if not keepdims: expvar = expvar.squeeze(axis=tuple(feature_axis))

    return expvar


def align_subspaces(basis1, basis2, feature_axis=-2, component_axis=-1, return_cosines=False,
                    keepdims=True, _compute_uv=True):
    """
    Optimally align bases of two subspaces using Bjorck & Golub principal angles method.

    Computes new bases that optimally align subspaces, essentially by computing the canonical
    correlation between them (SVD of their matrix product).

    Also optionally returns the cosines of the angles between the aligned subspaces, in ranked
    order (decreasing cosines/increasing angles).

    Note that if you are only interested in the principal angles/cosines themselves, you should use
    :func:`.subspace_principal_angles` instead.

    Parameters
    ----------
    basis1 : ndarray, shape=(n_features,n_components1) or (...,n_features,...,n_components1,...)
        Basis matrix for first subspace to align. Column vectors (along `component_axis`)
        correspond to sequence of orthonormal basis vectors for subspace, eg the top-k eigenvectors
        from a PCA; features along each column typically correspond to different neural
        channels/units.

        Can be single basis matrix or stack of such matrices in an array (eg for different
        experimental conditions, timepoints, etc.), with the array axis corresponding to different
        vector features (eg channels) given in `feature_axis`.


    basis2 : ndarray, shape=(n_features,n_components2) or (...,n_features,...,n_components2,...)
        Basis for second subspace to align. Often, this is derived from some other condition
        (eg timepoint, exp condition) that we are aligning/comparing the basis1 condition to.

        May have different number of components (basis vectors), but must otherwise have same shape
        as `basis1`.

    feature_axis : int, default: -2 (second-to-last axis)
        Axis of `basis1` and `basis2` corresponding to different features.
        This will typically correspond to different neural channels/units.

    component_axis : int, default: -1 (last axis)
        Axis of `basis1` and `basis2` corresponding to different basis/eigenvectors of each basis.
        This corresponds to different components derived from a dimensionality reduction analysis.

    return_cosines : bool, default: False
        If True, returns third output = cosines of principal angles btwn subspace.
        If False, just returns aligned bases.

    keepdims : bool, default: True
        If True, retains reduced `feature_axis` as length-one axes in output.
        If False, removes reduced `feature_axis` from outputs.

    Returns
    -------
    basis1 : ndarray, shape=(n_features,n_components1) or (...,n_features,...,n_components1,...)
        Optimally aligned basis for first subspace. Same shape as input.

    basis2 : ndarray, shape=(n_features,n_components2) or (...,n_features,...,n_components2,...)
        Optimally aligned basis for second subspace. Same shape as input.

    cosines : ndarray, shape=(k,) or (...,[1,]...,k,...), optional
        Series of angle cosines between each pair of basis vectors in compared subspaces. Length of
        returned angles is the minimum of the number of vectors/components in basis1 and basis2:
        k = min(n_components1,n_components2)

        Returns a 1D array for single matrix inputs. For stacked matrices, returns array the same
        shape as `basis1/2` but with length of `component_axis` = min(n_components1,n_components2)
        and `feature_axis` reduced to length 1 (if `keepdims` is True) or removed (if `keepdims`
        is False).

    References
    ----------
    Bjorck & Golub (1973) Math Comp http://dx.doi.org/10.2307/2005662 (original method)
    """
    s1 = [s for s in basis1.shape if s != component_axis]
    s2 = [s for s in basis2.shape if s != component_axis]
    assert (basis1.ndim == basis2.ndim) and (s1 == s2), \
        "Subspace bases `basis1` and `basis2` must have same shape (except for `component_axis`)"

    # Single matrix in standard (n_features,n_components) format
    if basis1.ndim == 2:
        if (feature_axis not in [0,-2]) or (component_axis not in [1,-1]):
            basis1 = basis1.T
            basis2 = basis2.T
    # Stack of matrixes (or non-standard transpose)
    else:
        basis1,shape1,ndim1 = standardize_array_3d(basis1,
                                                   axis1=feature_axis, axis2=component_axis,
                                                   target_axis1=-2, target_axis2=-1)
        basis2,shape2,ndim2 = standardize_array_3d(basis2,
                                                   axis1=feature_axis, axis2=component_axis,
                                                   target_axis1=-2, target_axis2=-1)

    # Compute SVD of matrix product of subspace bases U,S,V = svd(basis1.T * basis2)
    # -> Singular vector matrixes U,V = projection matrixes to compute new optimally-aligned bases
    # -> singular values diagonal(S) = angle cosines
    # U,V ~ ([...,]n_features,k), S ~ ([...,]k);  k = min(n_features,n_components)
    extra_args = dict(full_matrices=False, compute_uv=_compute_uv)
    if basis1.ndim == 2:
        svd_out = np.linalg.svd(basis1.T @ basis2, **extra_args)
    else:
        svd_out = np.linalg.svd(_transpose_end(basis1) @ basis2, **extra_args)

    if _compute_uv:
        U,_,Vt = svd_out
        V = Vt.T if basis1.ndim == 2 else _transpose_end(Vt)  # svd outputs V.T; Convert to V.
        basis1_aligned = basis1 @ U
        basis2_aligned = basis2 @ V
    else:
        cosines = svd_out
        basis1_aligned = None
        basis2_aligned = None

    # Deal w/ floating point error -- limit range strictly to [-1,+1]
    if return_cosines: cosines = np.clip(cosines, -1, 1)

    if basis1.ndim == 2:
        if (feature_axis not in [0,-2]) or (component_axis not in [1,-1]):
            if return_cosines: cosines = cosines.T
            if _compute_uv:
                basis1_aligned = basis1_aligned.T
                basis2_aligned = basis2_aligned.T
    else:
        if return_cosines:
            cosines = cosines[...,np.newaxis,:] # Replace reduced feature axis with singleton
            cosines = undo_standardize_array_3d(cosines, shape1, ndim1,
                                                axis1=feature_axis, axis2=component_axis,
                                                target_axis1=-2, target_axis2=-1)
        if _compute_uv:
            basis1_aligned = undo_standardize_array_3d(basis1_aligned, shape1, ndim1,
                                                       axis1=feature_axis, axis2=component_axis,
                                                       target_axis1=-2, target_axis2=-1)
            basis2_aligned = undo_standardize_array_3d(basis2_aligned, shape2, ndim2,
                                                       axis1=feature_axis, axis2=component_axis,
                                                       target_axis1=-2, target_axis2=-1)

    if not keepdims and (basis1.ndim > 2):
        if return_cosines:
            cosines = cosines.squeeze(axis=feature_axis)

        if _compute_uv:
            basis1_aligned = basis1_aligned.squeeze(axis=feature_axis)
            basis2_aligned = basis2_aligned.squeeze(axis=feature_axis)

    if return_cosines:  return basis1_aligned, basis2_aligned, cosines
    else:               return basis1_aligned, basis2_aligned


# =============================================================================
# Subspace comparison/alignment/overlap functions
# =============================================================================
def subspace_reconstruction_index(data_comp, basis_ref, basis_comp, feature_axis=-1, sum_axis=None,
                                  expansion_basis_comp=None, expansion_basis_ref=None,
                                  keepdims=True):
    """
    Compute index of similarity betweeen two subspaces estimated from distinct "reference"
    and "comparison" sets of data, quantified as the proportion of variance in the comparison data
    captured by cross-projecting it onto the reference subspace, normalized by the proportion of
    comparison data variance captured by projecting it onto a subspace estimated from itself::

        index = (1 - ||X_comp - X_comp * W_ref * W_ref.T||^2 / ||X_comp||^2) /
                (1 - ||X_comp - X_comp * W_comp * W_comp.T||^2 / ||X_comp||^2)

    where X's are the data, W's are the subspace bases, and ||.|| is the L2 norm

    This index could be used to compare subspaces between, for example, different timepoints or
    experimental condition. Note that the reference and comparison subspaces need not have the
    dimensionality (number of components).

    The index ranges from 0 (no alignment) to 1 (perfect alignment). Note that inputs are treated
    asymmetrically. Roughly, "reference" and "comparison" here correspond to "training" and
    "testing" conditions for decoding (classification, regression) analysis. In cross-temporal
    decoding terms, this index would correspond to normalizing the cross-temporal matrix by its
    columns (normalizing each test time by its own decoder fit).

    Function allows mass-multivariate computation of index for multiple data vectors
    (eg different trials/conditions, time points, etc.), but only for a single subspace basis.

    This is the "subspace overlap" index of Russo 2020. Note: In Russo's formulation, sums
    of squares are Frobenius (matrix) norms across features (channels/neurons) AND time.
    By default, this function only computes L2 (vector) norms across features (neurons).
    To match Russo, set `sum_axis` = <time axis> (or to sum across trials and neurons, set
    `sum_axis` = <trial axis>). Also, according Russo 2020's formula (p. e4, "Methods:
    Quantification and Statistical analysis : Subspace overlap"), they use the *unsquared*
    norm; we instead use the *squared* norm, to keep it analogous to the univariate formulas
    for Sums of Squares.

    Parameters
    ----------
    data_comp : ndarray, shape=(n_features,) or (...,n_features,...)
        "Comparison" condition data array, used to compute variance captured by cross-projection
        onto reference basis `basis_ref` derived from a different condition.

        Can be single data vector (typically corresponding to a population of neural
        channels/units) or stack of such vectors (eg for different trials, timepoints, etc.),
        with the array axis corresponding to different vector features given in `feature_axis`.

    basis_ref : ndarray, shape=(n_features,n_components_ref)
        Basis matrix for subspace estimated from "reference" condition, which is used to reduce
        data dimensionality down to `n_component_ref`. Column vectors correspond to sequence of
        orthonormal basis vectors for subspace, eg the top-k eigenvectors from a PCA; features
        along each column correspond to same feature in `data` (eg neural channels/units).
        For simplicity, only a single basis is allowed as input.

    basis_comp : ndarray, shape=(n_features,n_components_comp)
        Basis matrix for subspace estimated from "comparison" condition subspace.
        Shape and interpretation otherwise same as `basis_ref`; doesn't need to have same number
        of components (dimensionality) as `basis_ref`.

    feature_axis : int, default: -1 (last axis)
        Axis of `data` corresponding to different features.
        This will typically correspond to different neural channels/units.

    sum_axis : int, default: None (no other axis)
        Optional additional axis (in addition to `feature_axis`) to compute Sums of Squares, and
        thus explained variance along (ie using matrix/Frobenius norm along two dims instead of
        standard L2/Euclidean norm along feature_axis). In Russo 2020 formulation, this would
        correspond to the time axis; for trial-based data, this might correspond to the trial axis.

    expansion_basis_comp : ndarray, shape=(n_components_comp,n_features), default: basis_comp.T
        Basis matrix for re-expansion of dimensionality-reduced data back to original data
        dimensionality, for comparison subspace. Only needed for dimensionality reduction methods
        (eg dPCA) that estimate distinct bases for dimensionality reduction and re-expansion
        (the "decoder" and "encoder" axes, respectively, of Kobak et al 2016).

        Otherwise, this defaults to the transpose of the subspace `basis` (as is the case for
        PCA, LDA, and most standard dimensionality reduction methods).

    expansion_basis_ref : ndarray, shape=(n_components_ref,n_features), default: basis_ref.T
        As above, but for reference subspace.

    keepdims : bool, default: False
        If True, retains reduced `feature_axis` as length-one axis in output.
        If False, removes reduced `feature_axis` from outputs.

    Returns
    -------
    overlap : ndarray, shape=float or (...,[1,]...)
        Subspace overlap index, measuring overlap btwn reference and comparison subspaces.

        Returns a float for single vector input. For stacked vectors, returns array the same
        shape as `data` but with `feature_axis` (and `sum_axis` if set) reduced to length 1
        (if `keepdims` is True) or removed (if `keepdims` is False).

    References
    ----------
    - Russo ... M.Churchland (2020) Neuron https://doi.org/10.1016/j.neuron.2020.05.020
    """
    kwargs_den = dict(feature_axis=feature_axis, sum_axis=sum_axis,
                      expansion_basis=expansion_basis_ref, keepdims=keepdims)
    kwargs_num = dict(feature_axis=feature_axis, sum_axis=sum_axis,
                      expansion_basis=expansion_basis_comp, keepdims=keepdims)

    return (subspace_reconstruction_var(data_comp, basis_ref, **kwargs_den) /
            subspace_reconstruction_var(data_comp, basis_comp, **kwargs_num))


def subspace_projection_index(cov_comp, basis_ref, basis_comp, feature_axis=(-2,-1),
                              keep_components=False, keepdims=True, **kwargs):
    """
    Compute index of similarity betweeen two subspaces estimated from distinct "reference"
    and "comparison" sets of data, quantified as the magnitude of the cross-projection of the
    comparison data onto the reference subspace, normalized by the self-projection of the
    comparison subspace onto its own subspace::

        index = trace(W_ref.T * C_comp * W_ref) / trace(W_comp.T * C_comp * W_comp)

    where C's are the data covariance matrices and W's are the subspace bases

    This index could be used to compare subspaces between, for example, different timepoints or
    experimental condition. Note that the reference and comparison subspaces need not have the
    dimensionality (number of components).

    The index ranges from 0 (no alignment) to 1 (perfect alignment). Note that inputs are treated
    asymmetrically. Roughly, "reference" and "comparison" here correspond to "training" and
    "testing" conditions for decoding (classification, regression) analysis. In cross-temporal
    decoding terms, this index would correspond to normalizing the cross-temporal matrix by its
    columns (normalizing each test time by its own decoder fit).

    Function allows mass-multivariate computation of index for multiple data vectors
    (eg different trials/conditions, time points, etc.), but only for single reference and
    comparison subspace bases.

    This is (close to) the "subspace alignment" index of Elsayed 2016 and Yoo 2020. The difference
    is for the "self-variance" in the denominator, they simply compute the sum of the n_components
    eigenvalues. We instead use the same projection formula as in the numerator, because it allows
    you to compute an internally cross-validated `basis_comp` (ie arguably the proper way to make
    this comparison). It also simplifies the input arguments. If you don't cross-validate, you
    should get the same result as the Elsayed formulation (but also you are biasing results in
    favor of the self-projection case / against the cross-projection case, as you are training
    and testing on the same "comparison" data, ie "double-dipping" or doing a "circular analysis").

    Parameters
    ----------
    cov_comp : ndarray, shape=(n_features,n_features) or (...,n_features,n_features,...)
        Cross-feature (eg cross-channel/unit) covariance matrix of "comparison" condition data.

        Can be single covariance matrix (typically corresponding to covariance of a population of
        neural channels/units) or stack of such matrixes (eg for different trials, timepoints,
        etc.), with the pair of array axes corresponding to different vector features
        (eg channels/neurons) given in `feature_axis`.

    basis_ref : ndarray, shape=(n_features,n_components_ref)
        Basis matrix for subspace estimated from "reference" condition, which is used to reduce
        data dimensionality down to `n_component_ref`. Column vectors correspond to sequence of
        orthonormal basis vectors for subspace, eg the top-k eigenvectors from a PCA; features
        along each column correspond to same feature in `data` (eg neural channels/units).
        For simplicity, only a single basis is allowed as input.

    basis_comp : ndarray, shape=(n_features,n_components_comp)
        Basis matrix for subspace estimated from "comparison" condition subspace.
        Shape and interpretation otherwise same as `basis_ref`. Only needs to have same number
        of components (dimensionality) as `basis_ref` if `keep_components` is True.

    feature_axis : array-like, shape=(2,), default: (-2,-1) (last two array axes)
        Axes of `cov` corresponding to different data vector features. Must have 2 values.
        This will typically correspond to covariance between different neural channels/neurons.

    keep_components : bool, default: False
        If True, returns variance explained by each basis component vector (eg PC) separately.
        If False, return total (summed) variance explained by all basis vectors.

    keepdims : bool, default: True
        If True, retains reduced `feature_axis` as length-one axes in output.
        If False, removes reduced `feature_axis` from outputs.

    **kwargs
        All other keyword args passed directly to :func:`.subspace_projection_var`

    Returns
    -------
    alignment : float or ndarray, shape=(...,[1,1,],...)
        Subspace alignment index, measuring alignment btwn reference and comparison subspaces.

        Returns a float for single matrix input. For stacked matrixes, returns array the same
        shape as `basis` but with `feature_axis` reduced to length 1 (if `keepdims` is True)
        or removed (if `keepdims` is False). If `keep_components` is True, returns array the
        same shape as `data`.

    References
    ----------
    - Elsayed ... Cunningham (2016) Nature Comms https://doi.org/10.1038/ncomms13239
    - Yoo & Hayden (2020) Neuron https://doi.org/10.1016/j.neuron.2019.11.013 (use in cognition)
    """
    if keep_components:
        assert basis_ref.shape[1] == basis_comp.shape[1], \
            ValueError("If you want to retain values for each component in output, ref and comp"
                       "bases must have same number of components (%d vs %d)" %
                       (basis_ref.shape[1], basis_comp.shape[1]))

    # Note: Any normalization will be same for numerator,denominator, so don't bother
    kwargs.update(dict(feature_axis=feature_axis, keep_components=keep_components,
                       normalization='none', keepdims=keepdims))

    return (subspace_projection_var(cov_comp, basis_ref, **kwargs) /
            subspace_projection_var(cov_comp, basis_comp, **kwargs))


def subspace_error_index(basis_ref, basis_comp, feature_axis=-2, component_axis=-1,
                         keepdims=True):
    """
    Compute index of similarity betweeen two subspaces estimated from distinct "reference"
    and "comparison" sets of data, quantified as 1 - the magnitude (L2 norm) of the
    cross-projection of the comparison subspace onto the nullspace of the reference subspace,
    normalized by the norm of the comparison subspace::

        error = ||(I - W_ref * inv(W_ref.T * W_ref) * W_ref.T) * W_comp|| / ||W_comp||
        index = 1 - error

    where W's are the subspace bases, and ||.|| is the L2 matrix (Frobenius) norm over
    the feature and component axes

    This is 1 - the "normalized subspace error" index of Gokcen 2022. We compute 1 - their index,
    converting their measure of error into a measure of alignment. The resulting measure ranges
    from 0 to 1. A value of 0 indicates the column space of the reference subspace lies completely
    with the null space of the comparison subspace, and thus the comparison captures no component
    of the reference.

    Parameters
    ----------
    basis_ref : ndarray, shape=(...,n_features,...,n_components_ref,...)
        Basis matrix for subspace estimated from "reference" condition (ie which is used to reduce
        data dimensionality down to `n_component_ref`). Column vectors correspond to sequence of
        orthonormal basis vectors for subspace, eg the top-k eigenvectors from a PCA.

        Can be single basis matrix (by default, shape=(n_features,n_components_ref)) or stack of
        such matrixes (eg for different experimental conditions, timepoints, etc.), with the array
        axis corresponding to different features (eg channels/neurons) given in `feature_axis` and
        the axis corresponding to subspace basis components given in `component_axis`.

    basis_comp : ndarray, shape=(...,n_features,...,n_components_comp,...)
        Basis matrix for subspace estimated from "comparison" condition subspace.
        Shape and interpretation otherwise same as `basis_ref`; doesn't need to have same number
        of components (dimensionality) as `basis_ref`.

    feature_axis : int, default: -2 (second-to-last axis)
        Axis of `basis_ref` and `basis_comp` corresponding to different features.
        This will typically correspond to different neural channels/units.
        By default, second-to-last axis of array of stacked matrices (1st axis of single matrix).

    component_axis : int, default: -1 (last axis)
        Axis of `basis_ref/comp` corresponding to different subspace components/dimensions.
        By default, last axis of array of stacked matrices (2nd axis of single matrix).

    keepdims : bool, default: True
        If True, retains reduced `feature_axis` and `component_axis` as length-one axis in output.
        If False, removes reduced `feature_axis` and `component_axis` from outputs.

    **kwargs
        All other keyword args passed directly to :func:`.subspace_projection_var`

    Returns
    -------
    overlap : ndarray, shape=float or (...,[1,]...)
        Subspace overlap index, measuring overlap btwn reference and comparison subspaces.

        Returns a float for single vector input. For stacked vectors, returns array the same
        shape as `data` but with `feature_axis` (and `sum_axis` if set) reduced to length 1
        (if `keepdims` is True) or removed (if `keepdims` is False).

    References
    ----------
    Gokcen ... Yu (2022) Nature CompSci https://doi.org/10.1038/s43588-022-00282-5 eqn. 9
    """
    # Special case of 1D bases -- expand with singleton component dim, set axis params approp.
    inputs_1d = (basis_ref.ndim == 1) or (basis_comp.ndim == 1)
    if inputs_1d:
        if basis_ref.ndim == 1:     basis_ref = basis_ref[:,np.newaxis]
        if basis_comp.ndim == 1:    basis_comp = basis_comp[:,np.newaxis]
        feature_axis = 0
        component_axis = 1

    s_ref = [s for s in basis_ref.shape if s != component_axis]
    s_comp= [s for s in basis_ref.shape if s != component_axis]
    assert (basis_ref.ndim == basis_comp.ndim) and (s_ref == s_comp), \
        "Subspace bases `basis_ref`and `basis_comp` must have same shape (except for `component_axis`)"

    if feature_axis < 0: feature_axis = basis_ref.ndim + feature_axis
    if component_axis < 0: component_axis = basis_ref.ndim + component_axis

    n_features = basis_ref.shape[feature_axis]

    # Stack of bases: Move feature and component axes to end of data array (axes -2 and -1, resp.)
    if basis_ref.ndim > 2:
        W_comp,_,_ = standardize_array_3d(basis_comp, axis1=feature_axis, axis2=component_axis,
                                          target_axis1=-2, target_axis2=-1, reshape=False)
        W_ref,shape,ndim = standardize_array_3d(basis_ref, axis1=feature_axis, axis2=component_axis,
                                                target_axis1=-2, target_axis2=-1, reshape=False)
    # Single matrix: Ensure feature and component axes are 1st and 2nd axes, resp.
    elif (feature_axis == 1) and (component_axis == 0):
        W_comp = basis_comp.T
        W_ref = basis_ref.T
    else:
        W_comp = basis_comp
        W_ref = basis_ref

    # Compute nullspace of reference space ~ ([...,]n_features,n_features)
    W_ref_T = W_ref.T if W_ref.ndim == 2 else _transpose_end(W_ref)
    W_ref_null = np.eye(n_features) - W_ref @ np.linalg.pinv(W_ref_T @ W_ref) @ W_ref_T

    # Frobenius (matrix) norm across features and components of
    #  reference nullspace * comparison column space ~ ([...,]n_features,n_components_comp)
    num = np.linalg.norm(W_ref_null @ W_comp, axis=(-2,-1), keepdims=True)
    # Frobenius (matrix) norm across features and components of comparison subspace matrix
    den = np.linalg.norm(W_comp, axis=(-2,-1), keepdims=True)
    # Final index is 1 - normalized error ~ ([...,]n_features,n_components_comp)
    index = 1 - (num / den)

    # If stacked basis matrixes input, reshape output to original data shape
    if basis_ref.ndim > 2:
        index = undo_standardize_array_3d(index, shape, ndim,
                                          axis1=feature_axis, axis2=component_axis,
                                          target_axis1=-2, target_axis2=-1)
        if not keepdims: index = index.squeeze(axis=(feature_axis,component_axis))
        elif inputs_1d: index = index.squeeze(axis=component_axis)
    else:
        index = index.item()    # Extract scalar index from array

    return index


def subspace_principal_angles(basis1, basis2, feature_axis=-2, component_axis=-1, output='radian',
                              keepdims=True):
    """
    Compute principal angles between subspaces as a measure of subspace similarity/overlap.

    These are the angles between subspaces, in increasing order of angle (decreasing order of
    their cosines), derived from a canonical correlation analysis btwn the two subspace bases
    (SVD of their matrix product). This essentially finds the optimal projection matrixes to
    for the two bases to minimize the angle between them; these minimized angles are returned
    in `angles`.

    Note that, unlike other subspace similarity indexes, inputs here are treated symmetrically --
    principal_angle(basis1,basis2) = principal_angle(basis2,basis1). Also, unlike other indexes,
    this one is vector-valued for each subspace comparison, with one principal angle per subspace
    component (PC/eigenvector).

    Function only returns principal angles. To also get aligned subspaces bases that minimize
    principal angles (optimally align subspaces), use :func:`.align_subspaces`.

    Function allows mass-multivariate computation of index for multiple data vectors
    (eg different trials/conditions, time points, etc.), but only for single reference and
    comparison subspace bases.

    Function allows mass-multivariate computation of principal angles for multiple subspaces
    (eg corresponding to different experimental conditions, time points, etc.).

    This method was developed originally by Bjorck & Golub 1973, and first used for neuroscience
    applications (AFAIK) by Gallego 2018.

    Parameters
    ----------
    basis1 : ndarray, shape=(n_features,n_components1) or (...,n_features,...,n_components1,...)
        Basis matrix for first subspace to align. Column vectors (along `component_axis`)
        correspond to sequence of orthonormal basis vectors for subspace, eg the top-k eigenvectors
        from a PCA; features along each column typically correspond to different neural
        channels/units.

        Can be single basis matrix or stack of such matrices in an array (eg for different
        experimental conditions, timepoints, etc.), with the array axis corresponding to different
        vector features (eg channels) given in `feature_axis`.

    basis2 : ndarray, shape=(n_features,n_components2) or (...,n_features,...,n_components2,...)
        Basis for second subspace to align. Often, this is derived from some other condition
        (eg timepoint, exp condition) that we are aligning/comparing the basis1 condition to.

        May have different number of components (basis vectors), but must otherwise have same shape
        as `basis1`.

    output : {'radian','degree','cosine'}, default: 'radian'
        Type of "angle" to output: Angles in radians (range [0,pi]) or degrees (range [0,180]),
        or the angle cosines (range [-1,+1]).

    feature_axis : int, default: -2 (second-to-last axis)
        Axis of `basis1` and `basis2` corresponding to different features.
        This will typically correspond to different neural channels/units.

    component_axis : int, default: -1 (last axis)
        Axis of `basis1` and `basis2` corresponding to different basis/eigenvectors of each basis.
        This corresponds to different components derived from a dimensionality reduction analysis.

    keepdims : bool, default: True
        If True, retains reduced `feature_axis` as length-one axes in output.
        If False, removes reduced `feature_axis` from outputs.

    Returns
    -------
    angles : ndarray, shape=(k,) or (...,[1,]...,k,...) (k = min(n_components1,n_components2))
        Series of angles between each matched pair of components (basis vectors) in the compared
        subspaces. Length of returned angles is the minimum of the number of components betweeen
        basis1 and basis2.

        Returns a 1D array for single matrix inputs. For stacked matrixes, returns array the same
        shape as `basis1/2` but with length of `component_axis` = min(n_components1,n_components2)
        and `feature_axis` reduced to length 1 (if `keepdims` is True) or removed (if `keepdims`
        is False).

    References
    ----------
    - Bjorck & Golub (1973) Math Comp http://dx.doi.org/10.2307/2005662 (original method)
    - Gallego et al. (2018) Nature Comms https://doi.org/10.1038/s41467-018-06560-z (use in neuro)
    - Tang et al. (2020) eLife https://doi.org/10.7554/eLife.58154 (use in PFC)
    """
    # Use principal angle algorithm to compute cosines of optimal principal angles btwn subspaces
    _,_,cosines = align_subspaces(basis1, basis2,
                                  feature_axis=feature_axis, component_axis=component_axis,
                                  return_cosines=True, keepdims=keepdims, _compute_uv=False)

    # Transform principal angle cosines to requested output type/units
    if 'cosin' in output:       return cosines
    elif 'radian' in output:    return np.arccos(cosines)
    elif 'degree' in output:    return np.rad2deg(np.arccos(cosines))
    else:
        raise ValueError("Unsupported option '%s' for `output`" % output)


# =============================================================================
# Cross-validation objects and functions
# =============================================================================
class OddEvenSplit(BaseCrossValidator):
    """
    Implements traditional odd/even trial split-half cross-validation.

    Depending on value of `cross` parameter, either implements single train/test split
    or two cross-validated split where odd and even swap train/test roles.

    Inherits from sklearn.model_selection.BaseCrossValidator, but is needed
    b/c this type of cv scheme is not implemented in sklearn.

    Parameters
    ----------
    cross : bool, default: False
        If False, generates a single split with odd trials for train and even trials for test.
        If True, generates two splits, one as above, and one with train/test roles reversed.
    """

    def __init__(self, cross=False):
        BaseCrossValidator.__init__(self)
        self.cross = cross

    def split(self, X, y=None):
        """
        Generate indices to split data into training and test set.

        X : ndarray, shape=(n_obs, n_features)
            Data array. Only used to get number of observations (trials).

        Any other parameters are ignored, just there to match sklearn API.

        Yields
        ------
        train_idxs : generator, shape=(n_splits,) of ndarray, shape=(n_obs/2,)
            Training set indices for split. Here, all odd or all even trials

        test_idxs : generator, shape=(n_splits,) of ndarray, shape=(n_obs/2,)
            Test set indices for split. Here, all even or all odd trials.
        """
        n = X.shape[0]
        # Note: "even/odd" here are defined by 1-offset trial #'s
        odd  = np.arange(0,n,2)
        even = np.arange(1,n,2)

        for i_split in range(self.get_n_splits()):
            yield (odd,even) if i_split == 0 else (even,odd)

    def get_n_splits(self, X=None, y=None):
        """ Return  the number of splitting iterations in the cross-validator """
        return 2 if self.cross else 1


class BalancedKFold(BaseCrossValidator):
    """
    Implements version of k-fold cross-validation with observations (trials) fully balanced
    between classes (conditions) within each fold.

    :func:`sklearn.model_selection.StratifiedKFold` does something similar, but permits slightly
    unbalanced classes within each fold in order to balance the total number of observations
    across folds.

    BalancedKFold forces all classes to be balanced within each fold, but permits the overall
    number of observations to vary slightly across folds.

    BalancedKFold should be used for applications where the number of observations (trials)
    must be exactly balanced across classes (conditions), such as dPCA.

    NOTE: All classes/conditions must be balanced in the original (unsplit) data

    Inherits from sklearn.model_selection.BaseCrossValidator, but is needed
    b/c this type of cv scheme is not implemented in sklearn.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into folds.
        Note that if shuffle=False, folds will follow the order of trials in each
        condition (ie fold1 will be the first n trials, fold2 the next n, etc.)

    random_state : int or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Pass an int for reproducible output across multiple function calls.
        Otherwise unrepeatable random sequences will generated.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        BaseCrossValidator.__init__(self)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """
        Generate indices to split data into training and test set.

        X : ndarray, shape=(n_obs, n_features)
            Data array. Not actually used here.

        y : array-like of shape (n_obs,)
            The target variable for supervised learning problems (labels).
            Balancing is done based on the y labels.

        Yields
        ------
        train_idxs : generator, shape=(n_splits,) of ndarray, shape=(n_train[split],)
            Training set indices for split.

        test_idxs : generator, shape=(n_splits,) of ndarray, shape=(n_test[split],)
            Test set indices for split.
        """
        n_splits = self.n_splits
        shuffle = self.shuffle
        random_state = self.random_state

        if random_state is not None: set_random_seed(random_state)

        # Find all conditions (classes) in y labels and number of trials per condition
        conds, n_per_cond = np.unique(y, return_counts=True)
        assert np.all(n_per_cond == n_per_cond[0]), "All classes (conditions) must have same n"
        n_per_cond = n_per_cond[0]

        # Ideal number of condition trials per cross-validation split. May be non-integer.
        n_per_split_ideal = n_per_cond / n_splits
        # Non-integer part of n_per_split_ideal
        n_per_split_rem = (n_per_cond % n_splits) / n_splits

        # (n_splits,) list with final number of trials per condition in each test split
        # These are integer-valued, but may differ between splits
        n_per_split = floor(n_per_split_rem*n_splits) * [ceil(n_per_split_ideal)] + \
                      ceil((1-n_per_split_rem)*n_splits) * [floor(n_per_split_ideal)]

        # Extract (n_conds,) list containing (n_per_cond,) lists of trials in each condition
        cond_trials = [(y == cond).nonzero()[0] for cond in conds]

        # Shuffle order of trials within each condition (so splits are not grouped by trial number)
        if shuffle:
            cond_trials = [this_cond_trials[np.random.permutation(n_per_cond)]
                           for this_cond_trials in cond_trials]

        for i_split in range(n_splits):
            # (n_per_split[i_split],) list of indexes into each cond list for current test split
            # Select next set of trials in n_per_split
            start = sum(n_per_split[:i_split])
            test_cond_idxs = range(start, start+n_per_split[i_split])

            # (n_per_cond-n_per_split[i_split],) list of indexes into each cond list for
            #  current train split. Select all trials not in current test split.
            train_cond_idxs = [idx for idx in range(n_per_cond) if idx not in test_cond_idxs]

            # Set list of actual train, test trials from indexes above
            train_trials = [this_cond_trials[train_cond_idxs] for this_cond_trials in cond_trials]
            test_trials = [this_cond_trials[test_cond_idxs] for this_cond_trials in cond_trials]

            # Convert to ndarrays and sort in ascending order, to match sklearn outputs
            train_trials = np.hstack(train_trials)
            test_trials = np.hstack(test_trials)
            train_trials.sort()
            test_trials.sort()

            yield (train_trials, test_trials)


    def get_n_splits(self, X=None, y=None):
        """ Return  the number of splitting iterations in the cross-validator """
        return self.n_splits


class DummyCrossValidator(BaseCrossValidator):
    """
    Implements a "dummy" cross-validation object that mirrors sklearn.model_selection
    "Splitter" cross-validator objects, but just uses all data for both training AND
    testing (ie, does not actually implement any cross-validation).

    Useful mainly just for simplifying code that allows cross-validation or not, so
    you don't have to write separate code blocks for both cases.
    """

    def __init__(self, **kwargs):
        BaseCrossValidator.__init__(self, **kwargs)

    def split(self, X, y=None):
        """
        Generates indices to split data into "training" and "test" set
        (which are actually the same here)

        Parameters
        ----------
        X : ndarray, shape=(n_obs, n_features)
            Data array. Only used to get number of observations (trials).

        Any other parameters are ignored, just there to match sklearn API.

        Yields
        ------
        train_idxs : generator, shape=(n_splits,) of ndarray, shape=(n_obs,)
            Training set indices for split. Here, all observations (trials).

        test_idxs : generator, shape=(n_splits,) of ndarray, shape=(n_obs,)
            Test set indices for split. Here, all observations (trials).
        """
        yield np.arange(X.shape[0]), np.arange(X.shape[0])

    def get_n_splits(self, X=None, y=None):
        """ Return the number of splitting iterations in the cross-validator """
        return 1


# =============================================================================
# Helper functions
# =============================================================================
def _transpose_end(data):
    """ Transpose final two axes of data array """
    axes = np.arange(data.ndim)
    axes[-2],axes[-1] = axes[-1],axes[-2]
    return data.transpose(tuple(axes))
