"""
Parametric tests of "face validity" for multivariate analyses in multi.py

Data simulation functions
^^^^^^^^^^^^^^^^^^^^^^^^^
- biased_random_subspaces : Generate random subspaces biased toward covariance structure of data
"""
import os
import warnings
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits import mplot3d
from scipy.linalg import orth
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from spynal.utils import set_random_seed, iarange, randperm, condition_mean
# from spynal.utils import is_symmetric, is_positive_definite
from spynal.plots import plot_line_with_error_fill
from spynal.multi import covariance_matrix, dimensionality, \
    subspace_reconstruction_var, subspace_projection_var, \
    subspace_reconstruction_index, subspace_projection_index, subspace_error_index, \
    subspace_principal_angles

warnings.filterwarnings("error",category=RuntimeWarning)

# =============================================================================
# Simulated data generators (TODO Clean these up and move some of these to somewhere public-facing?)
# =============================================================================
def simulate_dimensional_data(n_dims=8, method='random', n_conds=16, n_chnls=100, n_trials=50,
                              gain=1.0, noise_sd=0.0, offset=0.0, dims=None, shuffle_labels=False,
                              seed=None):
    """ Simulate random data of given dimensionality """
    assert method in ['random','factorial'], "Unsupported value '%s' input for `method`" % method
    # DEL if method == 'factorial':
    #     assert n_conds == 16, "Can only run n_conds=16 for method='factorial' (%d input)" % n_conds

    if seed is not None: set_random_seed(seed)

    # Generate matrix of noiseless single-rep mean data of given dimensionality ~ (n_conds,n_chnls)
    # Generate mean data as matrix multiplication of normal rand samples with implicit dimension
    if method == 'random':
        # # Generate random full n-dimensional orthonormal basis ~ (n_chnls,n_chnls)
        # U,_ = np.linalg.qr(np.random.randn(n_chnls,n_chnls))
        # # Generate random latent weights ~ (n_chnls,n_conds)
        # Z = np.random.randn(n_chnls,n_conds)
        # eigenspectrum = 'elbow'
        # if eigenspectrum == 'elbow': s = np.r_[np.ones((n_dims,)), 0.2*np.ones((n_chnls-n_dims,))]
        # # Combine together and transpose ~ (n_conds,n_chnls)
        # X = (U @ np.diag(s) @ Z).T

        # # Generate random n_dims-dimensional orthonormal basis ~ (n_chnls,n_dims)
        # U,_ = np.linalg.qr(np.random.randn(n_chnls,n_dims))
        # # Generate random latent weights ~ (n_dims,n_conds)
        # Z = np.random.randn(n_dims,n_conds)
        # eigenspectrum = 'power'
        # if eigenspectrum == 'flat':     s = np.ones((n_dims,))
        # elif eigenspectrum == 'power':  s = iarange(1,n_dims).astype(float)**(-0.1)
        # # Combine together and transpose ~ (n_conds,n_chnls)
        # X = (U @ np.diag(s) @ Z).T

        # Generate random n_dims-dimensional orthonormal basis ~ (n_chnls,n_dims)
        U,_ = np.linalg.qr(np.random.randn(n_chnls,n_dims))
        # Generate random latent weights ~ (n_dims,n_conds)
        Z = np.random.randn(n_dims,n_conds)
        # Combine together and transpose ~ (n_conds,n_chnls)
        X = (U @ Z).T

        # X = np.random.randn(n_conds,n_dims) @ np.random.randn(n_dims,n_chnls)

    # Generate mean data as weights on factorial decomposition of 16-cond design
    else:
        # Create (n_conds,n_factors) orthogonal design matrix
        if n_conds == 2:
            design = [[+1, 1],
                      [-1, 1]]
        elif n_conds == 4:
            design = [[+1,  1,  1, 1],
                      [-1,  1, -1, 1],
                      [+1, -1, -1, 1],
                      [-1, -1,  1, 1]]
        elif n_conds == 16:
            design = [[+1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                      [-1,  1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1],
                      [+1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1],
                      [-1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
                      [+1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1],
                      [-1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1,  1],
                      [+1, -1, -1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1,  1,  1],
                      [-1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1],
                      [+1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,  1],
                      [-1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1],
                      [+1, -1, -1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1,  1,  1,  1],
                      [-1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1],
                      [+1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,  1,  1],
                      [-1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1],
                      [+1, -1, -1, -1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1,  1],
                      [-1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1]]
        else:
            raise ValueError("n_dims=%d not supported (yet)" % n_dims)
        design = np.asarray(design)

        # Set random weights for all factors and channels ~ (n_factors=n_conds,n_chnls)
        # Note: Need to use signed weights (both positive and negative)
        weights = np.random.randn(n_conds,n_chnls)    # Normal RVs
        # ALT weights = (np.random.binomial(1, 0.5, size=(n_conds,n_chnls))*2) - 1    # Random signs

        # Randomly select `n_dims` factors from factorial model
        sel_factors = randperm(n_conds,n_dims) if dims is None else dims
        weights *= np.in1d(np.arange(n_conds),sel_factors)[:,np.newaxis]

        # Matrix multiply design matrix by channel factor weights to get single-rep means
        X = design @ weights

    # Simulate final data means as gain * X + offset ~ (n_conds,n_chnls)
    means = gain * X + offset

    # Simulate data by replicating means n times and adding Gaussian noise ~ (n_conds*n_trials,n_chnls)
    data = np.tile(means, (n_trials,1)) + \
        np.random.normal(loc=0, scale=noise_sd, size=(n_conds*n_trials,n_chnls))

    labels = np.tile(np.arange(n_conds), n_trials)
    if shuffle_labels: labels = labels[np.random.permutation(n_conds*n_trials)]

    return data, labels


def rotate_around_axis(vectors, angle, axis='x'):
    """ Rotate 1/more 3D vectors around cardinal axis using rotation matrix """
    t = np.deg2rad(angle)

    if axis == 'x':
        rotMx = np.column_stack([[1,0,0], [0,cos(t),-sin(t)], [0,sin(t),cos(t)]])
    elif axis == 'y':
        rotMx = np.column_stack([[cos(t),0,sin(t)], [0,1,0], [-sin(t),0,cos(t)]])

    return rotMx @ vectors


def create_subspace(basis=None, angle=0, axis='x'):
    """
    Create basis for 2D subspace embedded within 3D space to use for simulating multivariate data

    Parameters
    ----------
    basis : array-like, shape=(3,2), default: [[1,1,0], [-1,1,0]]
        Two orthogonal 3D vectors that form 2D basis embedded within 3D space

    angle : float, default: 0
        Angle (degrees) to rotate defined `basis` around given `axis`

    axis : {'x','y'}, default: 'x'
        Axis to rotate basis around (if angle != 0): 'x'-axis or 'y'-axis

    Returns
    -------
    basis : ndarray, shape=(3,2)
        Orthonormal basis for 2D subspace within 3D space, optionally rotated as above
    """
    if basis is None:
        basis = np.column_stack(([1,1,0], [-1,1,0]))
    else:
        basis = np.asarray(basis)
        assert basis.shape == (3,2), "Basis must have shape=(3,2) (2 basis vectors in 3D space)"
        assert np.isclose(basis[:,0].dot(basis[:,1]), 0), "Basis vectors must be orthogonal"

    basis = basis / np.linalg.norm(basis,axis=0)
    if angle != 0: basis = rotate_around_axis(basis, angle, axis=axis)
    return basis


def simulate_subspace_dataset(basis=None, n_conds=2, plane_angle=45, gain=10.0, sd=5.0, n=100, seed=None):
    """
    Parameters
    ----------
    n_conds : int, default: 2
        Number of distinct conditions/groups to simulate

    n : int, default: 100
        Number of trials/observations to simulate *per condition*

    Returns
    -------
    data : ndarray, shape=(n*n_conds,n_chnls)
        Simulated data for multiple repetitions of one/more conditions.

    labels : ndarray, shape=(n*n_conds,), dtype=int
        Condition/group labels for each trial. Sorted in group order to simplify visualization.
    """
    if seed is not None: set_random_seed(seed)
    if basis is None:   basis = create_subspace()
    else:               basis = np.asarray(basis)

    # Set angle of each condition mean in subspace = cond1 angle + even sampling of 360 btwn conds
    angles = plane_angle + np.arange(0,360,360/n_conds)
    thetas = np.deg2rad(angles)[:,np.newaxis]   # -> (n_conds,1) in rads

    # Compute condition means *within* subspace = [gain*cos(angle), gain*sin(angle)] ~ (n_conds,2=(x,y))
    means = gain * np.column_stack([np.cos(thetas), np.sin(thetas)])

    # Project cond means from subspace into full 3D space ~ (n_conds,3)
    means = means @ basis.T

    # For each condition, simulate condition values as multivariate normal ~ (n,3)
    data = []
    labels = []
    for cond in range(n_conds):
        # TODO Code up actual covariance matrix
        cond_data = np.random.multivariate_normal(means[cond,:], sd*np.eye(3), size=(n,))
        data.append(cond_data)
        labels.append(cond*np.ones((n,)))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    return data, labels


def plot_simulated_data(data, labels, basis=None, basis2=None):
    """ Plot simulated data and underlying basis in 3D scatter """
    conds = np.unique(labels)

    lim = np.max(np.abs(data))
    lim = (-lim,lim)
    ax = plt.subplot(1,1,1, projection='3d', xlim=lim, ylim=lim, zlim=lim)
    ax.tick_params(axis='both', direction='in')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.plot3D((lim[0],lim[-1]), (0,0), (0,0), '-', color=[0.75,0.75,0.75], linewidth=0.5)
    ax.plot3D((0,0), (lim[0],lim[-1]), (0,0), '-', color=[0.75,0.75,0.75], linewidth=0.5)

    for i_cond,cond in enumerate(conds):
        cond_trials = labels == cond
        ax.plot3D(data[cond_trials,0], data[cond_trials,1], data[cond_trials,2], 'o',
                  markersize=4, linewidth=1.5, alpha=0.3)

    if basis is not None:
        for i_comp in range(basis.shape[1]):
            color = [1,0,0] if i_comp == 0 else [0,0,1]
            if basis2 is not None:
                ax.plot3D([-10*basis2[0,i_comp],10*basis2[0,i_comp]],
                          [-10*basis2[1,i_comp],10*basis2[1,i_comp]],
                          [-10*basis2[2,i_comp],10*basis2[2,i_comp]], ':', color=color, linewidth=1)

            ax.plot3D([-10*basis[0,i_comp],10*basis[0,i_comp]],
                      [-10*basis[1,i_comp],10*basis[1,i_comp]],
                      [-10*basis[2,i_comp],10*basis[2,i_comp]], '-', color=color, linewidth=1)


def simulate_paired_multivariate_datasets(n_latents=3, latent_weight=None,
                                          var_weight=(1,1), cov_weight=(0.5,0.5),
                                          n_chnls=(100,100), n_samples=200, seed=None):
    """
    Simulate paired multivariate-correlated datasets/populations using probabilistic CCA model.

    Useful, for example, for testing CCA and related cross-decomposition methods.

    Based on formulas in Supplementary Note "Characterizing changes in the interaction structure"
    in Semedo 2022 paper below. Each population is simulated as multivariate normal, but both are
    driven by activation of `n_latents` latent dimensions. Relative weight/strength of driving of
    each population by latent dimensions can be set using `latent_weight`. Relative weight of
    within-population variance/covariance can set using `var_weight`/`cov_weight`.

    Notation:

    - q = n_latents = Number of latent dimensions driving both populations
    - p_x,p_y = n_chnls[0],n_chnls[1] = number of channels/units in each population
    - z = latent_weight = (q,) vector of latent dim weights that influence activity of both pops
    - W_x,W_y = (p_x,q),(p_y,q) mapping matrices from latents to each population's activity
    - Psy_x,Psy_y = (p_x,p_x),(p_y,p_y) within-population covariance matrix for each population

    Parameters
    ----------
    n_latents : int, default: 3
        Number of latent dimensions driving both populations. 'q' in Semedo 2022.

    latent_weight : array-like, shape=(q,), dtype=float, default=np.ones((q,))
        Strength of each latent dimension in driving population activity, relative to
        within-population variance/covariance. Should have one value
        per latent dimension, in range [0,Inf]. Implemented via scaling of latent dimensions.

        This is not explicitly part of the pCCA formulation in Semedo, but could be
        absorbed into columnar scaling of the W_x,W_y mapping matrices.

    var_weight : array-like, shape=(2,), dtype=float, default=(1,1)
        Strength of each within-population variance (diagonals of Psy_x,Psy_y matrices),
        relative to across-population latent. One value per population, in range [0,Inf].
        Variance for each channel/unit is selected from normal(0,var_weight) distribution.

    cov_weight : array-like, shape=(2,), dtype=float, default=(0.5,0.5)
        Strength of each within-population covariance (off-diagonals of Psy_x,Psy_y matrices),
        relative to across-population latent, in range [0,Inf]. Covariance for each channel/unit
        pair is selected from normal(0,cov_weight) distribution.

    n_chnls : array-like, shape=(2,), dtype=int, default=(100,100)
        Number of channels/neurons to simulate in each population. (p_x,p_y) in Semedo 2022.

    n_samples : int, default=200
        Number of independent random samples to generate from pCCA model. Can be thought of as
        independent trials or distinct timepoints.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Returns
    -------
    X : ndarray, shape=(n_samples,p_x)
        Random simulated activity across all channels in population 1, based on latent structure.
        `n_samples` random samples are generated.

    Y : ndarray, shape=(n_samples,p_y)
        Random simulated activity across all channels in population 2, based on latent structure

    References
    ----------
    Semedo ... Yu (2022) Nature Comms (Supp. Note) https://doi.org/10.1038/s41467-022-28552-w
    """
    if seed is not None: set_random_seed(seed)
    if latent_weight is None: latent_weight = np.ones((n_latents,))
    elif np.isscalar(latent_weight) and (n_latents > 1): latent_weight *= np.ones((n_latents,))
    if np.isscalar(var_weight): var_weight *= np.ones((2,))
    if np.isscalar(cov_weight): cov_weight *= np.ones((2,))

    # Take square root of var/cov weight bc they get squared below to create pos-def matrix
    var_weight = np.sqrt(np.asarray(var_weight))
    cov_weight = np.sqrt(np.asarray(cov_weight))

    # Generate random vector of latents from standard normal distribution ~ (q,)
    z = np.random.normal(0, 1, size=(n_latents,))
    # Weight latents by requested weights
    z = latent_weight * z

    # Parts below are not explicitly done in Semedo 2022 formulation, but these need to be set
    #  somehow, so I'm just setting them as normal RV's with 0 mean, given variance.
    # Not 100% sure this is the best way to generate them...

    # Generate random mapping matrices to map latents onto neural activity ~ (p_x,q) and (p_y,q)
    W_x = np.random.normal(0, 1, size=(n_chnls[0],n_latents))
    W_y = np.random.normal(0, 1, size=(n_chnls[1],n_latents))

    # Generate random variances and covariances
    var_x = np.random.normal(0, var_weight[0], size=(n_chnls[0],))
    var_y = np.random.normal(0, var_weight[1], size=(n_chnls[1],))

    cov_x = np.random.normal(0, cov_weight[0], size=(n_chnls[0],n_chnls[0]))
    cov_y = np.random.normal(0, cov_weight[1], size=(n_chnls[1],n_chnls[1]))

    # Remove diagonal from covariance matrixes, replace it with variances ~ (p_x,p_x) and (p_y,p_y)
    cov_x -= np.diag(np.diag(cov_x))
    cov_y -= np.diag(np.diag(cov_y))

    cov_x += np.diag(var_x)
    cov_y += np.diag(var_y)

    # Multiply each matrix and its transpose to generate symmetric covariance matrixes
    # These are the final within-population covariance matrices Psy_x and Psy_y
    Psy_x = cov_x @ cov_x.T
    Psy_y = cov_y @ cov_y.T

    # assert is_symmetric(Psy_x) and is_symmetric(Psy_y), \
    #     ValueError("`covariance` must be symmetric matrix")
    # assert is_positive_definite(Psy_x, semi=True) and is_positive_definite(Psy_y, semi=True), \
    #     ValueError("`covariance` must be positive semi-definite matrix")

    # Simulate neural activity for each population as multivariate normal with
    #  mean ~ mapping matrix * latent, and covariance ~ within-population covariance
    #  for multiple independent samples -> X ~ (n_samples,p_x), Y ~ (n_samples,p_y)
    X = np.random.multivariate_normal(W_x @ z, Psy_x, size=(n_samples,))
    Y = np.random.multivariate_normal(W_y @ z, Psy_y, size=(n_samples,))

    return X, Y


def random_subspace_pair(n, k, angles=None, degrees=False, seed=None):
    """
    Generate a pair of random subspace basis matrices, optionally with given angles
    between each of their component dimensions.

    Useful for testing measures of subspace alignment/overlap.

    Algorithm generates one k-dimensional subspace and its orthogonal complement.
    The second subspace is generated as the linear combination of these two, with
    the weights based on the cosine and and sine of the angles, respectively.

    Parameters
    ----------
    n : int
        Full ambient dimensionality of space to simulate subspace within.
        Corresponds to number of channels/units in actual neural data.

    k : int
        Subspace dimensionality. Number of component dimensions of both simulated subspaces.

    angles : float or ndarray, shape=(k,), default: None (random subspaces)
        Angle(s) between corresponding dimensions of simulated subspaces. Given in radians, unless
        `degrees` is True.

        If single scalar value, that same angle is used for all subspace dimensions.
        If `angles` is None, two random subspaces without any defined geometric relationship
        are generated.

    degrees : bool, default: False (angles in radians)
        Set=True, if `angles` are given in degrees, rather than radians

    seed : int, default: None (unseeded random numbers)
        Random generator seed for repeatable results

    Returns
    -------
    S1, S2 : ndarray, shape=(n,k)
        Pair of k-dimensional subspace bases (within n-dimensional ambient space), optionally
        with given angles between their corresponding component dimensions
    """
    # TODO Add docstring. Factor this into a few subfuncs, eg random_basis(), random_rotation()?
    assert k <= n, ValueError("k must be <= n (input: n=%d,k=%d)" % (n,k))
    if seed is not None: set_random_seed(seed=seed)

    if angles is not None:
        assert k <= n/2, \
            ValueError("If angles are input, k must be <= n/2 (input: n=%d,k=%d)" % (n,k))
        angles = np.atleast_1d(angles)

        # Expand (copy) scalar angle to length k
        if len(angles) == 1:
            angles = angles*np.ones((k,))
        else:
            assert len(angles) == k, \
                ValueError("`angles` must be scalar or length-k (%d vs %d)" % (len(angles), k))
        if degrees: angles = np.deg2rad(angles)

        # Diagonal matrices of cosines and sines of `angles`
        cos_theta = np.diag(np.cos(angles))
        sin_theta = np.diag(np.sin(angles))

    # Generate a random orthogonal basis for the first subspace
    S1 = orth(np.random.randn(n, k))  # n-dimensional space, k-dimensional subspace

    # If angles given, generate second basis with given angles from first
    if angles is not None:
        # Compute orthonormal basis for the orthogonal complement of S1
        Q_full, _ = np.linalg.qr(np.hstack((S1, np.eye(n))), mode='reduced')
        S1_ortho = Q_full[:,k:]
        assert k >= S1_ortho.shape[1], \
            ValueError("Insufficient dimensions in the orthogonal complement to construct S2")
        # Take first k components of orthogonal complement of S1
        S1_ortho = S1_ortho[:,:k]

        # DEL? Not sure why this part is necessary and algo seems to work fine w/o it
        # # Generate a random orthogonal matrix Q with orthonormal columns
        # Q_rand = np.random.randn(k, k)
        # Q, _ = np.linalg.qr(Q_rand)
        # S2 = S1 @ cos_theta + S1_ortho @ Q @ sin_theta

        # Construct S2 as linear combination of S1 and S1_ortho (with weights based on theta)
        # DEL print(S1.shape, cos_theta.shape, S1_ortho.shape, sin_theta.shape)
        S2 = S1 @ cos_theta + S1_ortho @ sin_theta
        # Orthonormalize S2
        S2, _ = np.linalg.qr(S2)

    # If no angles given, just generate another random orthogonal basis for the second subspace
    else:
        S2 = orth(np.random.randn(n, k))  # n-dimensional space, k-dimensional subspace

    return S1, S2


def biased_random_subspaces(data1, data2=None, n_components=None, n_resamples=10000,
                            seed=None, **kwargs):
    """
    Generate Monte Carlo random subspaces biased toward covariance structure of given data.

    Eigenvectors/eigenvalues of data are computed and paired with random samples of
    Gaussian-valued coefficient vectors, to generate random subspaces with a bias toward
    data covariance.

    Data may be single data array or two data arrays pooled together (as you would do
    for a shuffle test that randomized labels across contrasted data arrays).

    Method from Elsayed et al. 2016 (Supplementary Note 3), where it was used to generate a
    null distribution of subspace alignment based on comparison of different instances of the
    same subspace + noise. This was used to show two distinct subspaces estimated from real data
    are more orthogonal than expected by chance if they were two samplings of the same subspace.

    TODO generalize to mass-multivariate with obs/feature_axis args?

    Parameters
    ----------
    data1 : ndarray, shape=(n_obs1,n_features)
        Data whose covariance is used to generate random subspaces based on. If no `data2` is
        input, random subspaces are based solely on the structure of `data1`.

        First axis is observations, usually trials (or conditions for condition mean data).
        Second axis is multivariate features for each observation, usually neural channels/units.

    data2 : ndarray, shape=(n_obs2,n_features), default: None (just use data1)
        If input, `data1` and `data2` are pooled together (concatenated across observation
        axis), and the covariance of the pooled data is used to generate biased random subspaces.

        This is generally used when measuring some relationship (eg subspace alignment) between
        data1 and data2, and you would like to measure the relationship expected by chance.

        Feature axis must be same length as `data1`, but observation axis may be different.

    n_components: int, default: min(n_obs_total,n_features)
        Number of components to extract in eigendecomposition. Defaults to minimum of n_features
        and n_observations (pooled across `data1` and `data2` if both input).

    n_resamples : int, default: 10000
        Number of independent random resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    **kwargs :
        Any other keyword args passes directly to :func:`.covariance_matrix`

    Yields
    ------
    v_align : generator, shape=(n_resamples,) of [ndarray, shape=(n_features,n_components)]
        Generator to iterate over for significance test or confidence interval computation.

        Each iteration contains a Monte Carlo random resampling of a subspace based on the
        covariance structure of `data1` (and `data2` as well, if input)

    References
    ----------
    Elsayed,...Churchland,Cunningham (2016) Nat Comms https://doi.org/10.1038/ncomms13239
    (see Supplementary Note 3)
    """
    if seed is not None: set_random_seed(seed)
    if data2 is not None:   data = np.concatenate((data1,data2), axis=0)
    else:                   data = data1
    n_obs = data.shape[0]
    n_features = data.shape[1] if data.ndim > 1 else 1
    n_components_max = min(n_obs,n_features)
    if n_components is None: n_components = n_components_max
    else:
        assert n_components <= n_components_max, \
            "Requested `n_components` (%d) exceeds max allowed due to data size (%d)" % \
            (n_components, n_components_max)

    # Compute covariance matrix of data ~ (n_features,n_features)
    C = covariance_matrix(data, **kwargs)

    # Compute eigenvalues S ~(n_features,)
    # and eigenvectors U ~ (n_features,n_features) of data covariance matrix
    S,U = np.linalg.eigh(C)
    # HACK Rectify tiny negative values (due to floating point error) to zero
    S = np.clip(S, a_min=0, a_max=None)
    # Take square root of eigenvalues and expand to ~(n_features,n_features) diagonal matrix
    S_sqrt = np.diag(np.sqrt(S))

    # Iterate thru multiple random resamples, yielding a `v_align` each iteration
    for _ in range(n_resamples):
        # Generate set of Gaussian random vectors
        v = np.random.standard_normal((n_features,n_components))

        # Numerator = U*sqrt(S)*v ~ (n_features,n_components)
        num = U @ S_sqrt @ v
        # Denominator = Frobenius norm (matrix 2-norm) of numerator ~ scalar
        denom = np.linalg.norm(num)

        # Compute orthonormal basis of (num/den) as its left singular vectors
        # ~ (n_features,n_components). This is "orth(Z)" in Elsayed 2016.
        v_align,_,_ = np.linalg.svd(num/denom, full_matrices=False)

        yield v_align


# =============================================================================
# Validity tests
# =============================================================================
def test_dimensionality(method, test='dimension', test_values=None, n_reps=5, do_coding_dim=True,
                        seed=None, do_tests=True, do_plots=False, plot_dir=None, **kwargs):
    """ Run tests on dimensionality estimation functions """
    if not do_coding_dim:
        assert method != 'pc_noise', "PC noise method only works for coding dimensionality"
    if seed is not None: set_random_seed(seed)

    n_conds = kwargs.pop('n_conds',16)
    sim_args = dict(n_trials=kwargs.pop('n_trials', 50),
                    n_chnls=kwargs.pop('n_chnls', 100),
                    gain=kwargs.pop('gain', 1.0),
                    noise_sd=kwargs.pop('noise_sd', 0.1),
                    offset=kwargs.pop('offset', 0.0),
                    n_dims=kwargs.pop('n_dims',8),
                    n_conds=n_conds,
                    method=kwargs.pop('sim_method','random'),
                    shuffle_labels=kwargs.pop('shuffle_labels',False))

    # Set up arguments/data generators for each test type
    if test in ['n_dims','dimension']:
        test_values = [1,2,4,8,16] if test_values is None else test_values # ALT iarange(1,n_conds)
        del sim_args['n_dims']
        gen_data = lambda n_dims: simulate_dimensional_data(n_dims=n_dims, **sim_args)

    elif test in ['n','n_trials']:
        test_values = [10,20,50,100,200,500] if test_values is None else test_values
        del sim_args['n_trials']
        gen_data = lambda n_trials: simulate_dimensional_data(n_trials=n_trials, **sim_args)

    elif test == 'gain':
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['gain']
        gen_data = lambda gain: simulate_dimensional_data(gain=gain, **sim_args)

    elif test in ['noise_sd','spreads','sd']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['noise_sd']                    # Delete preset arg so uses arg to lambda below
        gen_data = lambda noise_sd: simulate_dimensional_data(noise_sd=noise_sd, **sim_args)

    elif test == 'n_conds':
        test_values = [2,4,8,16,32,64] if test_values is None else test_values
        del sim_args['n_conds']                     # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_conds: simulate_dimensional_data(n_conds=n_conds, **sim_args)

    elif test == 'n_chnls':
        test_values = [50,100,200,500,1000] if test_values is None else test_values
        del sim_args['n_chnls']                     # Delete preset arg so uses arg to lambda below
        gen_data = lambda n_chnls: simulate_dimensional_data(n_chnls=n_chnls, **sim_args)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)


    results = np.empty((len(test_values),n_reps))

    for i_value,test_value in enumerate(test_values):
        print("TRUE DIM = ", test_value)
        for i_rep in range(n_reps):
            # Generate new set of synthetic data ~ (n_trials*n_conds, n_chnls)
            test_data, labels = gen_data(test_value)

            if do_coding_dim:
                results[i_value,i_rep] = dimensionality(test_data, labels, method=method, **kwargs)
            else:
                results[i_value,i_rep] = dimensionality(test_data, None, method=method, **kwargs)

    # Compute mean and std dev across different reps of simulation
    means   = results.mean(axis=1)
    sds     = results.std(axis=1,ddof=0)

    if do_plots:
        plt.figure()
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.errorbar(test_values, means, sds, marker='o')
        if test in ['n_dims','dimension']:
            plt.plot((test_values[0],test_values[-1]), (test_values[0],test_values[-1]), '-',
                     color='k', linewidth=0.5)
        plt.xlabel(test)
        plt.ylabel("Dimensionality (%s)" % method)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'multi-dim-summary-%s-%s' % (method,test)))

    # Determine if test actually produced the expected values
    # 'n_dims' : Test if stat increases monotonically with dimensionality of data
    if test in ['n_dims','dimension']:
        evals = [((np.diff(means) >= 0).all(),
                 "Values don't increase monotonically with data dimensionality")]
    # 'n'/'gain'/'noise_sd' : Test if stat is ~ constant (unbiased) across tested values
    elif test in ['n','gain','noise_sd']:
        evals = [(means.ptp() < sds.max(),
                 "Values have larger than expected range across %s" % test)]

    else:
        evals = []

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        # elif not cond:  print("Warning: " + message)

    return means, sds, passed


# TEMP ,'shatter'
def dimensionality_test_battery(methods=('pcnoise'),
                                tests=('n_dims','n','gain','noise_sd','n_conds','n_chnls'),
                                do_tests=True, **kwargs):
    """
    Run a battery of given tests on given dimensionality computation methods

    Parameters
    ----------
    methods : array-like of str, default: ('pcnoise')
        List of dimensionality computation methods to test.

    tests : array-like of str, default: ('n_dims')
        List of tests to run.

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    **kwargs :
        Any other kwargs passed directly to test_subspace_comparison()
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]

    # plot_dir = kwargs['plot_dir']
    for test in tests:
        for method in methods:
            print("Running %s test on %s" % (test,method))

            _,_,passed = test_dimensionality(method, test=test, do_tests=do_tests, **kwargs)

            print('%s' % 'PASSED' if passed else 'FAILED')
            # If saving plots to file, let's not leave them all open
            if 'plot_dir' in kwargs: plt.close('all')

            # TEMP
            # for noise_method in ['expvar','quantile','sd']:
            #     for which_eigs in ['all','first']:
            #         if (noise_method == 'expvar') and (which_eigs == 'first'): continue
            #         _,_,passed = test_dimensionality(method, test=test, do_tests=do_tests, noise_method=noise_method, which_eigs=which_eigs, **kwargs)
            #         plt.savefig(os.path.join(plot_dir,'multi-dim-summary-%s-%s-%s-%s.png' % (test,method,noise_method,which_eigs)))

            #         print('%s' % 'PASSED' if passed else 'FAILED')
            #         # If saving plots to file, let's not leave them all open
            #         if 'plot_dir' in kwargs: plt.close('all')
            # TEMP

            # _,_,passed = test_dimensionality(method, test=test, do_tests=do_tests, **kwargs)

            # print('%s' % 'PASSED' if passed else 'FAILED')
            # # If saving plots to file, let's not leave them all open
            # if 'plot_dir' in kwargs: plt.close('all')


def test_subspace_comparison_random(method, test='dim_ratio', test_values=None,
                                    n_dims=250, n_components=3, dim_ratio=0.5, angles=None, n_trials=100,
                                    n_resamples=1000, seed=None, do_tests=True, do_control=True,
                                    do_plots=False, plot_dir=None, **kwargs):
    """ Run tests on subspace expvar and comparison functions with random data """
    if seed is not None: set_random_seed(seed)

    # DEL sim_args = dict(eigs=kwargs.pop('eigs',None), n_resamples=n_resamples)
    sim_args = dict(angles=angles, degrees=True, seed=None)
    noise = kwargs.pop('noise',0)

    if test == 'n_dims':
        if test_values is None: test_values = [2,4,10,50,100,250,500]
        gen_spaces = lambda n_dims: random_subspace_pair(n_dims, n_components, **sim_args)
    elif test == 'n_dims_constant_ratio':
        if test_values is None: test_values = [2,4,10,50,100,250,500]
        gen_spaces = lambda n_dims: random_subspace_pair(n_dims, min(round(dim_ratio*n_dims), n_dims-1), **sim_args)
    elif test == 'n_components':
        if test_values is None: test_values = [2,4,10,50,100,250,500]
        gen_spaces = lambda n_components: random_subspace_pair(n_dims, n_components, **sim_args)
    elif test == 'dim_ratio':
        if test_values is None:
            test_values = iarange(0.1,0.9,0.1) if angles is None else iarange(0.1,0.5,0.1)
        gen_spaces = lambda ratio: random_subspace_pair(n_dims, min(round(ratio*n_dims), n_dims-1), **sim_args)
    elif test == 'angles':
        del sim_args['angles']
        if test_values is None: test_values = iarange(0,90,15)
        gen_spaces = lambda angles: random_subspace_pair(n_dims, n_components, angles=angles, **sim_args)
    elif test == 'noise':
        if test_values is None: test_values = [0.1,0.2,0.5,1,2]
        gen_spaces = lambda _: random_subspace_pair(n_dims, n_components, **sim_args)
    else:
        raise ValueError("Unsupported test type '%s'" % test)

    variable_components = (method == 'subspace_principal_angles') and \
                          (test in ['dim_ratio','n_dims_constant_ratio'])

    sample_data = do_control or (method in ['subspace_reconstruction_var','subspace_reconstruction_index'])
    # For principal angles method, #output values per call = n_components
    # (which is variable for dim_ratio test). 1 value per call for all other methods
    if method == 'subspace_principal_angles':
        if test == 'dim_ratio':
            values_per_call = np.max([min(round(test_value*n_dims), n_dims-1)
                                      for test_value in test_values])
        elif test == 'n_dims_constant_ratio':
            values_per_call = np.max([min(round(test_value*dim_ratio), test_value-1)
                                      for test_value in test_values])
        else:
            values_per_call = n_components
    else:
        values_per_call = 1
    results = np.full((len(test_values),n_resamples,values_per_call), fill_value=np.nan)
    if do_control:
        results_rand = np.full((len(test_values),n_resamples,values_per_call,10), fill_value=np.nan)

    for i_value,test_value in enumerate(test_values):
        if test == 'n_dims':
            n_dims = test_value
        elif test == 'n_dims_constant_ratio':
            n_dims = test_value
            n_components = min(round(dim_ratio*n_dims), n_dims-1)
        elif test == 'dim_ratio':
            n_components = min(round(test_value*n_dims), n_dims-1)
        elif test == 'noise':
            noise = test_value


        # # Create 2 generators with distinct sets of simulated random subspaces
        # subspaces1 = gen_spaces(test_value)
        # subspaces2 = gen_spaces(test_value)
        # print(n_dims, test_value, n_components)

        # for i_rsmp,(basis1,basis2) in enumerate(zip(subspaces1,subspaces2)):
        for i_rsmp in range(n_resamples):
            basis1, basis2 = gen_spaces(test_value)

            # Compute covariance matrix from subspace basis2, for methods that want it
            if method in ['subspace_projection_var','subspace_projection_index']:
                cov2 = basis2 @ basis2.T
                if noise != 0:
                    noise_mx = noise*np.random.randn(n_dims,n_dims)
                    cov2 += (noise_mx + noise_mx.T)/2

            # Sample data from subspace basies, for methods that want it
            if sample_data:
                # Multiply each basis by random coefficients to create datapoints in each subspace
                shape = (n_components,n_trials) if n_trials > 1 else (n_components,)
                data1 = (basis1 @ np.random.randn(*shape)).T
                data2 = (basis2 @ np.random.randn(*shape)).T
                # Add random Gaussian white noise
                if noise != 0:
                    data1 += noise*np.random.randn(n_trials,n_dims)
                    data2 += noise*np.random.randn(n_trials,n_dims)

            # Compute subspace comparison metric on simulated bases/data
            if method == 'subspace_reconstruction_var':
                res = subspace_reconstruction_var(data2, basis1, sum_axis=0, **kwargs)
            elif method == 'subspace_projection_var':
                res = subspace_projection_var(cov2, basis1, **kwargs)
            elif method == 'subspace_reconstruction_index':
                res = subspace_reconstruction_index(data2, basis1, basis2, sum_axis=0, **kwargs)
            elif method == 'subspace_projection_index':
                res = subspace_projection_index(cov2, basis1, basis2, **kwargs)
            elif method == 'subspace_error_index':
                res = subspace_error_index(basis1, basis2, **kwargs)
            elif method == 'subspace_principal_angles':
                res = subspace_principal_angles(basis1, basis2, output='degree', **kwargs) # output='cosine'

            if variable_components:
                results[i_value,i_rsmp,:n_components] = res
            else:
                results[i_value,i_rsmp,:] = res

            # Sample two random subspaces biased to pooled covariance of basis1/2, and recompute
            # comparison metric on them ("biased random subspace" control from Elsayed 2016)
            if do_control:
                bases_rand1 = biased_random_subspaces(data1, data2=data2,
                                                      n_components=n_components, n_resamples=10)
                bases_rand2 = biased_random_subspaces(data1, data2=data2,
                                                      n_components=n_components, n_resamples=10)

                # Compute subspace comparison metric on simulated random control bases/data
                for i_rsmp_rand,(basis_rand1,basis_rand2) in enumerate(zip(bases_rand1,bases_rand2)):
                    if method in ['subspace_projection_var','subspace_projection_index']:
                        cov_rand2 = basis_rand2 @ basis_rand2.T
                        if noise != 0:
                            noise_mx = noise*np.random.randn(n_dims,n_dims)
                            cov_rand2 += (noise_mx + noise_mx.T)/2

                    if method in ['subspace_reconstruction_var','subspace_reconstruction_index']:
                        data_rand2 = (basis2 @ np.random.randn(*shape)).T
                        if noise != 0: data_rand2 += noise*np.random.randn(n_trials,n_dims)

                    if method == 'subspace_reconstruction_var':
                        res = subspace_reconstruction_var(data_rand2, basis_rand1, sum_axis=0, **kwargs)
                    elif method == 'subspace_projection_var':
                        res = subspace_projection_var(cov_rand2, basis_rand1, **kwargs)
                    elif method == 'subspace_reconstruction_index':
                        res = subspace_reconstruction_index(data_rand2, basis_rand1, basis_rand2, sum_axis=0, **kwargs)
                    elif method == 'subspace_projection_index':
                        res = subspace_projection_index(cov_rand2, basis_rand1, basis_rand2, **kwargs)
                    elif method == 'subspace_error_index':
                        res = subspace_error_index(basis_rand1, basis_rand2, **kwargs)
                    elif method == 'subspace_principal_angles':
                        res = subspace_principal_angles(basis_rand1, basis_rand2, output='degree', **kwargs) # output='cosine'

                    if variable_components:
                        results_rand[i_value,i_rsmp,:n_components,i_rsmp_rand] = res
                    else:
                        results_rand[i_value,i_rsmp,:,i_rsmp_rand] = res

    # Compute mean and std dev across different reps of simulation
    means   = results.mean(axis=1)
    sds     = results.std(axis=1,ddof=0)

    if do_control:
        means_rand  = results_rand.mean(axis=(1,-1))
        sds_rand    = results_rand.std(axis=(1,-1),ddof=0)

    if do_plots:
        values_per_plot = min(values_per_call,3)    # Only plot 1st 3 principal axes
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        for j in range(values_per_plot):
            plt.errorbar(test_values, means[:,j], sds[:,j], marker='o', color=colors[j])
            if do_control:
                plot_line_with_error_fill(test_values, means_rand[:,j], sds_rand[:,j],
                                          color=colors[j], linestyle=':')
        plt.ylim((-5,95) if method == 'subspace_principal_angles' else (-0.05,1.05))
        plt.xlabel(test)
        plt.ylabel(method)

        # for sp in [1,2]:
        #     plt.subplot(1,2,sp)
        #     plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        #     for j in range(values_per_plot):
        #         if sp == 1: plt.errorbar(test_values, means[:,j], sds[:,j], marker='o')
        #         else:       plt.plot(test_values, sds[:,j], marker='o')
        #     plt.xlabel(test)
        #     plt.ylabel('Means' if sp == 1 else 'SDs')
        #     if sp == 1: plt.title(method)

        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'multi-subspace-summary-%s-%s' % (method,test)))

    if do_control:  return means, sds, means_rand, sds_rand
    else:           return means, sds


def subspace_comp_rand_test_battery(methods=('subspace_reconstruction_var','subspace_projection_var','subspace_reconstruction_index',
                                             'subspace_projection_index','subspace_principal_angles'),
                                    n_dims=(4,10,50,100,200), dim_ratios=tuple(iarange(0.1,0.9,0.1)), **kwargs):

    for method in methods:
        for i_dim,n_dim in enumerate(n_dims):
            print("Running dim_ratio test on %dD space" % n_dim)

            means,sds = test_subspace_comparison_random(method, test='dim_ratio', test_values=dim_ratios,
                                                        n_dims=n_dim, do_plots=False, **kwargs)

            plt.figure()
            values_per_plot = 3 if method == 'subspace_principal_angles' else 1    # Only plot 1st 3 principal axes
            plt.subplots(len(n_dims),1,i_dim+1)
            plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
            for j in range(values_per_plot):
                plt.errorbar(dim_ratios, means[:,j], sds[:,j], marker='o')
            plt.xlabel('Dimension ratio (k/n)')
            plt.title(method)
            # if plot_dir is not None:
            #     plt.savefig(os.path.join(plot_dir,'multi-subspace-summary-%s-%s' % (method,test)))


def test_subspace_comparison(method, test='basis_rot_angle', test_values=None,
                             basis_method='ideal', n_reps=100, seed=None,
                             do_tests=True, do_plots=False, plot_dir=None, **kwargs):
    """ Run tests on subspace expvar and comparison functions with parametrically varied data """
    if seed is not None: set_random_seed(seed)

    # Set up dim redux object to estimate basis from simulated data
    if basis_method.lower() in ['pca','grouppca']:
        dim_redux = PCA(n_components=2)
    elif basis_method.lower() == 'lda':
        dim_redux = LinearDiscriminantAnalysis(n_components=2)


    # Set up arguments to subspace and data simulation functions
    #   Override defaults with any simulation-related params passed to function
    #   Duplicate any scalar values to pair for data1,data2
    subspace_args = dict(angle=0, axis='x') # , basis=np.column_stack(([1,1,0], [-1,1,0])))
    for arg in subspace_args:
        if arg in kwargs: subspace_args[arg] = kwargs.pop(arg)
        if np.isscalar(subspace_args[arg]): subspace_args[arg] = [subspace_args[arg]]*2
    subspace_ref_args = {k:v[0] for k,v in subspace_args.items()}
    subspace_cmp_args = {k:v[1] for k,v in subspace_args.items()}

    sim_args = dict(n_conds=2, plane_angle=45, gain=10.0, sd=5.0, n=100)
    for arg in sim_args:
        if arg in kwargs: sim_args[arg] = kwargs.pop(arg)
        if np.isscalar(sim_args[arg]): sim_args[arg] = [sim_args[arg]]*2
    sim_ref_args = {k:v[0] for k,v in sim_args.items()}
    sim_cmp_args = {k:v[1] for k,v in sim_args.items()}

    basis_ref = create_subspace(**subspace_ref_args)
    data_ref,labels = simulate_subspace_dataset(basis=basis_ref, **sim_ref_args)
    data_ref -= data_ref.mean(axis=0, keepdims=True)
    # cov_ref = covariance_matrix(data_ref)
    if basis_method != 'ideal':
        if basis_method.lower() == 'grouppca':  X = condition_mean(data_ref,labels)[0]
        else:                                   X = data_ref
        dim_redux.fit(X=X, y=labels)    # Fit dim reduction model to simulated data

        if basis_method.lower() == 'lda':
            basis_ref = dim_redux.scalings_
        else:
            basis_ref = dim_redux.components_.T  # tranpose -> (n_features, n_components)

    # Set up arguments/data generators for each test type
    gen_subspace = lambda _: create_subspace(**subspace_cmp_args)
    if test == 'basis_rot_angle':
        test_values = iarange(0,90,5) if test_values is None else test_values
        del subspace_cmp_args['angle']          # Delete preset arg so uses arg to lambda below
        gen_subspace = lambda basis_rot_angle: create_subspace(**subspace_cmp_args, angle=basis_rot_angle)
        gen_data = lambda basis,_: simulate_subspace_dataset(basis=basis, **sim_cmp_args)
    elif test == 'plane_angle':
        test_values = iarange(0,180,5) if test_values is None else test_values
        del sim_cmp_args['plane_angle']          # Delete preset arg so uses arg to lambda below
        gen_data = lambda basis,plane_angle: simulate_subspace_dataset(basis=basis, plane_angle=plane_angle, **sim_cmp_args)
    elif test == 'dim_swap':
        test_values = [0,1]
        gen_subspace = lambda swap: (create_subspace(**subspace_cmp_args)[:,::-1] if swap else
                                     create_subspace(**subspace_cmp_args))
        gen_data = lambda basis,_: simulate_subspace_dataset(basis=basis, **sim_cmp_args)
    elif test == 'cond_sd':
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_cmp_args['sd']          # Delete preset arg so uses arg to lambda below
        gen_data = lambda basis,sd: simulate_subspace_dataset(basis=basis, sd=sd, **sim_cmp_args)
    elif test == 'cond_gain':
        test_values = [1,2,5,10,20,50] if test_values is None else test_values
        del sim_cmp_args['gain']          # Delete preset arg so uses arg to lambda below
        gen_data = lambda basis,gain: simulate_subspace_dataset(basis=basis, gain=gain, **sim_cmp_args)
    elif test == 'n_conds':
        test_values = iarange(2,12) if test_values is None else test_values
        del sim_cmp_args['n_conds']          # Delete preset arg so uses arg to lambda below
        gen_data = lambda basis,n_conds: simulate_subspace_dataset(basis=basis, n_conds=n_conds, **sim_cmp_args)
    elif test == 'n':
        test_values = [10,20,50,100,200,500] if test_values is None else test_values
        del sim_cmp_args['n']          # Delete preset arg so uses arg to lambda below
        gen_data = lambda basis,n: simulate_subspace_dataset(basis=basis, n=n, **sim_cmp_args)
    else:
        raise ValueError("Unsupported value '%s' input for `test`" % test)

    values_per_call = 2 if method == 'subspace_principal_angles' else 1
    results = np.empty((len(test_values),n_reps,values_per_call))

    for i_value,test_value in enumerate(test_values):
        # Generate subspace dimensions based on current test_value
        basis_cmp = gen_subspace(test_value)
        # print(test_value, sim_cmp_args['plane_angle'])
        # print(basis_cmp)
        # print(i_value, test_value); # print(basis_ref);  print(basis_cmp)

        for i_rep in range(n_reps):
            # Simulate data based on current test_value and compute its covariance
            data_cmp,labels = gen_data(basis_cmp, test_value)
            data_cmp -= data_cmp.mean(axis=0, keepdims=True)
            cov = covariance_matrix(data_cmp)

            # Use ideal basis used to simulate data as `basis` arg to comparison funcs
            if basis_method == 'ideal':
                basis_cmp_ = basis_cmp
            # Recompute basis by fitting a dimensionality reduction model to simulated data
            else:
                if basis_method.lower() == 'grouppca':  X = data_cmp
                else:                                   X = condition_mean(data_cmp,labels)[0]
                dim_redux.fit(X=X, y=labels)    # Fit dim reduction model to simulated data

                if basis_method.lower() == 'lda':
                    basis_cmp_ = dim_redux.scalings_
                else:
                    basis_cmp_ = dim_redux.components_.T  # tranpose -> (n_features, n_components)

            basis_ref_ = basis_ref
            basis_orig = basis_cmp_.copy()
            basis_ref_orig = basis_ref_.copy()

            # if i_rep == 1: print(basis_cmp_)
            if method == 'subspace_reconstruction_var':
                res = subspace_reconstruction_var(data_cmp, basis_ref_, sum_axis=0, **kwargs)
            elif method == 'subspace_projection_var':
                res = subspace_projection_var(cov, basis_ref_, **kwargs)
            elif method == 'subspace_reconstruction_index':
                res = subspace_reconstruction_index(data_cmp, basis_ref_, basis_cmp_, sum_axis=0, **kwargs)
            elif method == 'subspace_projection_index':
                res = subspace_projection_index(cov, basis_ref_, basis_cmp_, **kwargs)
            elif method == 'subspace_principal_angles':
                res = subspace_principal_angles(basis_ref_, basis_cmp_, output='cosine', **kwargs) # output='degree'

            assert (basis_orig == basis_cmp_).all() and (basis_ref_orig == basis_ref_).all(), "SHIT"
            results[i_value,i_rep,:] = res


    # Compute mean and std dev across different reps of simulation
    means   = results.mean(axis=1)
    sds     = results.std(axis=1,ddof=0)

    if do_plots:
        plt.figure()
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        for j in range(values_per_call):
            plt.errorbar(test_values, means[:,j], sds[:,j], marker='o')
        plt.xlabel(test)
        plt.ylabel(method)
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'multi-subspace-summary-%s-%s' % (method,test)))


    # Determine if test actually produced the expected values
    # 'basis_rot_angle' : Test if stat decreases monotonically with rot angle btwn ref and comp bases
    if test == 'basis_rot_angle':
        evals = [((np.diff(means[:,1]) >= 0).all() if method == 'subspace_principal_angles' else (np.diff(means) <= 0).all(),
                 "Values don't decrease monotonically with angle btwn bases")]
    # 'cond_sd' : Test if stat decreases monotonically with within-condition data std dev
    elif test == 'cond_sd':
        evals = [((np.diff(means[:,1]) >= 0).all() if method == 'subspace_principal_angles' else (np.diff(means) <= 0).all(),
                 "Values don't decrease monotonically with data variance")]
    # 'cond_gain' : Test if stat increases monotonically with between-condition gain (difference)
    elif test == 'cond_gain':
        evals = [((np.diff(means[:,1]) <= 0).all() if method == 'subspace_principal_angles' else (np.diff(means) >= 0).all(),
                 "Values don't increase monotonically with data gain")]
    # 'n_conds' : Test stat is ~ same for all values of n_conds
    elif test == 'n_conds':
        evals = [(means.ptp() <= sds.max(),
                 "Values have > expected range with different number of conds")]
    # 'n' : Test stat is ~ same for all values of n (unbiased by n)
    elif test in ['n','n_trials']:
        evals = [(means.ptp() <= sds.max(),
                 "Values have > expected range across n's (likely biased by n)")]
    else:
        evals = []

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  print("Warning: " + message)

    return means, sds, passed


def subspace_comp_test_battery(methods=('subspace_reconstruction_var','subspace_projection_var','subspace_reconstruction_index',
                                        'subspace_projection_index','subspace_principal_angles'),
                               tests=('basis_rot_angle','plane_angle','dim_swap','cond_sd','cond_gain','n','n_conds'), do_tests=True, **kwargs):
    """
    Run a battery of given tests on given subspace comparison methods

    Parameters
    ----------
    methods : array-like of str, default: ('subspace_reconstruction_var','subspace_projection_var','subspace_reconstruction_index',
                                           'subspace_projection_index','subspace_principal_angles')
        List of subspace comparison methods to test.

    tests : array-like of str, default: ('basis_rot_angle','cond_sd','cond_gain','n','n_conds')
        List of tests to run.

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    **kwargs :
        Any other kwargs passed directly to test_subspace_comparison()
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]

    for test in tests:
        for method in methods:
            print("Running %s test on %s" % (test,method))

            _,_,passed = test_subspace_comparison(method, test=test, do_tests=do_tests, **kwargs)

            print('%s' % 'PASSED' if passed else 'FAILED')
            # If saving plots to file, let's not leave them all open
            if 'plot_dir' in kwargs: plt.close('all')
