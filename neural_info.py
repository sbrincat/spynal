# -*- coding: utf-8 -*-
"""
neural_info     A module for computing measures of neural information about
                given task/behavioral variables. Computations are all mass-
                univariate across channels (units/LFPs/etc.).

FUNCTIONS
neural_info     Wrapper function computes neural information using any given method

### Area under receiver operating characteristic curve-related functions ###
auroc           Computes neural info as area under ROC curve
auroc_2groups   Computes area under ROC with two data distributions as args

### D-prime/Cohen's d-related functions ###
dprime          Computes neural information as d-prime
dprime_2groups  Computes d-rime with two data distributions as args

### Percent explained variance-related functions ###
pev             Computes neural info as percent explained variance
anova1          Computes PEV and stats using 1-way ANOVA
anova2          Computes PEV and stats using 2-way ANOVA
regress         Computes PEV and stats using 2-way linear regression

DEPENDENCIES
patsy           Python package for describing statistical models


Created on Mon Sep 17 00:05:25 2018

@author: sbrincat
"""
# TODO  Group2idx-type mechanism to convert arbitrary X to integer indexes in ANOVA models
# TODO  Should we default <groups> to be in alphanumeric order or in order of appearance in <labels>?

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import f as Ftest
from sklearn.linear_model import LinearRegression
from patsy import DesignMatrix, dmatrix


def neural_info(X, y, method='pev', **kwargs):
    """
    Wrapper function to compute mass-univariate neural information about
    some task/behavioral variable(s)

    info = neural_info(X,y,method='pev',**kwargs)

    ARGS
    X       (n_obs,n_terms) array-like. Design matrix.
            Must be same length as y.shape[0] along dimension DIM.

    y       (...,n_obs,...) ndarray. Neural data to fit. Axis <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels X.

    method  String. Method to use to compute information.
            Currently only have implemented 'pev' [default]

    **kwargs All other kwargs passed directly to information method function

    RETURNS
    exp_var (...,n_terms,...). Measure of information in y about X
            Shape is same as y, with observation axis reduced to length = n_terms.
    """
    method = method.lower()
    
    if method == 'pev':
        return pev(X,y,**kwargs)
    elif method in ['dprime','d','cohensd']:
        return dprime(X,y,**kwargs)
    elif method in ['auroc','roc','aucroc','auc']:
        return auroc(X,y,**kwargs)
    elif method in ['mutual_information','mutual_info']:
        return mutual_information(X,y,**kwargs)
    else:
        raise ValueError("Information method '%s' is not yet supported" % method)


# =============================================================================
# Mutual information analysis
# =============================================================================
def mutual_information(labels, data, axis=0, resp_entropy=None, groups=None):
    """
    Mass-univariate mutual information between set of responses 
    and binary "stimulus" (ie, 2 contrasted experimental conditions)
    
    info = mutual_information(labels,data,axis=0,resp_entropy=None,groups=None)

    Computes Shannon mutual information using standard equation (cf. Dayan & Abbott, eqn. 4.7): 
    I = H - Hnoise = -Sum(p(r)*log(p(r)) + Sum(p(cat)*p(r|s)*log(p(r|s)))

    where H = total response entropy, Hnoise = noise entropy, p(r) is response probability
    distribution, and p(r|s) is conditional probability of response given stimulus

    info = 0 indicates no mutual information between responses and experimental conditions
    info = 1 indicates maximum possible information btwn responses and conditions
    
    Computation is performed over trial/observation <axis>, in mass-univariate 
    fashion across data series in all other data dimensions (channels, time points, 
    frequencies, etc.).

    Note: This is a wrapper around mutual_information_2groups(), which accepts
    two data distributions as arguments, and is faster.

    ARGS
    labels      (n_obs,) array-like. List of group labels labelling observations from 
                each group. Should be only two groups represented, unless sub-selecting 
                two groups using <groups> argument.

    data        (...,n_obs,...) ndarray. Data values for both distributions to be compared.  

    axis        Scalar. Axis of data array to perform analysis on, corresponding
                to trials/observations. Default: first non-singleton dimension

    resp_entropy   (...,1,...). Total response entropy. Can optionally compute and input
                this to save repeated calculations (eg for distinct contrasts on same data)

    groups      (2,) array-like. Which group labels from <labels> to use. 
                Useful for computing info btwn pairs of groups in data with > 2 groups,
                Default: unique(labels) (all distinct values in <labels>)

    RETURNS
    info        (...,1,...) ndarray. Mutual information between responses 
                and experimental conditions (in bits). 
                
    REFERENCE     Dayan & Abbott, _Theoretical neuroscience_ ch. 4.1                
    """
    # TODO  Recode for > 2 groups
    labels = np.asarray(labels)
    
    # Find set of unique group labels in <labels>
    if groups is None: groups = np.unique(labels)

    assert len(groups) == 2, \
        "mutual_information: computation only supported for 2 groups (%d given)" % len(groups)

    # Reshape data array -> (n_observations,n_data_series) matrix
    data, data_shape = _reshape_data(data,axis)
    if resp_entropy is not None:
        resp_entropy,_ = _reshape_data(resp_entropy,axis)
    
    n_series = data.shape[1] if data.ndim > 1 else 1

    # Remove any observations not represented in <groups>
    idxs = (labels == groups[0]) | (labels == groups[1])
    if idxs.sum() != data.shape[0]:
        labels  = labels[idxs]
        data    = data[idxs,:]

    idxs1 = labels == groups[0]
    idxs2 = labels == groups[1]
    n1 = idxs1.sum()
    n2 = idxs2.sum()
    N  = n1 + n2

    assert (n1 != 0) and (n2 != 0), \
        "mutual_information: Data contains no observations (trials) for one or more groups"

    # Stimulus probabilities
    Ps1 = n1/N                  # P(stimulus==s1)
    Ps2 = n2/N                  # P(stimulus==s2)
            
    info = np.empty((1,n_series))
  
    for i_series in range(n_series):    
        data_series = data[:,i_series] if n_series > 1 else data.squeeze()

        # Compute total response entropy H (if not input)
        if resp_entropy is None:            
            # Response probability distributions
            _,counts    = np.unique(data_series, return_counts=True)
            Pr          = counts / N            # P(response==r)

            # Response entropy (Dayan & Abbott, eqn. 4.3)
            H   = _entropy(Pr)

        else:
            H   = resp_entropy[i_series]
        
        # Conditional response|stimulus probability distributions (eg, Prs1 is p(rateR|stim1))
        # Note: conditional probs of 0 are automatically eliminated, since by convention 0*log0=0
        _,counts    = np.unique(data_series[idxs1], return_counts=True)
        Prs1        = counts / counts.sum()                 # P(response==r | stimulus1)
        _,counts    = np.unique(data_series[idxs2], return_counts=True)
        Prs2        = counts / counts.sum()                 # P(response==r | stimulus2)
        
        # Conditional entropies for stimuli 1 & 2 (Dayan & Abbott, eqn. 4.5)
        Hs1     = _entropy(Prs1)
        Hs2     = _entropy(Prs2)
    
        # Noise entropy (Dayan & Abbott, eqn. 4.6)
        Hnoise  = Ps1*Hs1 + Ps2*Hs2

        # Mutual information is total response entropy - noise entropy (Dayan & Abbott, eqn. 4.7)
        info[i_series] = H - Hnoise

    info = _unreshape_data(info,data_shape,axis=axis)
        
    return info


    # Split data into 2 groups and use auroc_2groups() to actually do computation
    # return mutual_information_2groups(data.compress(labels == groups[0], axis=axis),
    #                      data.compress(labels == groups[1], axis=axis), 
    #                      axis=axis, signed=signed)


def mutual_information_2groups(data1, data2, axis=0, resp_entropy=None):
    """
    Mass-univariate mutual information between set of responses 
    and binary "stimulus" (ie, 2 contrasted experimental conditions)
        
    info = mutual_information_2groups(data1,data2,axis=0,resp_entropy=None)

    Computes Shannon mutual information using standard equation (cf. Dayan & Abbott, eqn. 4.7): 
    I = H - Hnoise = -Sum(p(r)*log(p(r)) + Sum(p(cat)*p(r|s)*log(p(r|s)))

    where H = total response entropy, Hnoise = noise entropy, p(r) is response probability
    distribution, and p(r|s) is conditional probability of response given stimulus

    info = 0 indicates no mutual information between responses and experimental conditions
    info = 1 indicates maximum possible information btwn responses and conditions
    
    Computation is performed over trial/observation <axis>, in mass-univariate 
    fashion across data series in all other data dimensions (channels, time points, 
    frequencies, etc.).
    
    Note: For convenience, mutual_information() is also provided as a wrapper around this
    function that accepts standard data,label arguments.
    
    ARGS
    data1/2     (...,n_obs1/n_obs2,...) ndarrays of arbitary size except for <axis>. 
                Sets of values for each of two distributions to be compared.  

    axis        Scalar. Dimension of data1,2 arrays to perform analysis on, corresponding
                to trials/observations. Default: first non-singleton dimension

    resp_entropy  (...,1,...). Total response entropy. Can optionally compute and input
                this to save repeated calculations (eg for distinct contrasts on same data)


    RETURNS
    info        (...,1,...) ndarray. Mutual information between responses 
                and experimental conditions (in bits). 
                
    REFERENCE     Dayan & Abbott, _Theoretical neuroscience_ ch. 4.1
    """
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]
    N  = n1 + n2

    assert (n1 != 0) and (n2 != 0), \
        "mutual_information: Data contains no observations (trials) for one or more groups"

    # Reshape data arrays -> (n_observations,n_data_series) matrix
    data1, data1_shape = _reshape_data(data1,axis)
    data2, data2_shape = _reshape_data(data2,axis)
    if resp_entropy is not None:
        resp_entropy,_ = _reshape_data(resp_entropy,axis)

    if data1.ndim > 1:
        n_series = data1.shape[1]
                    
        assert data1.shape[1] == data2.shape[1], \
            "mutual_information: Data distributions to compare must have same number of data series (timepts,freqs,channels,etc.) \
             (data1 ~ %d, data2 ~ %d)" % (data1.shape[1],data2.shape[1])
    else:
        n_series = 1
        
    # Stimulus probabilities
    Ps1 = n1/N                  # P(stimulus==s1)
    Ps2 = n2/N                  # P(stimulus==s2)
            
    info = np.empty((1,n_series))
  
    for i_series in range(n_series):    
        data1_series = data1[:,i_series] if n_series > 1 else data1.squeeze()
        data2_series = data2[:,i_series] if n_series > 1 else data2.squeeze()

        # Compute total response entropy H (if not input)
        if resp_entropy is None:
            # Concatenate data from both conditions together, and find all unique data values
            data_pooled = np.hstack((data1_series, data2_series))
            
            # Response probability distributions
            _,counts    = np.unique(data_pooled, return_counts=True)
            Pr          = counts / N            # P(response==r)

            # Response entropy (Dayan & Abbott, eqn. 4.3)
            H   = _entropy(Pr)

        else:
            H   = resp_entropy[i_series]
        
        # Conditional response|stimulus probability distributions (eg, Prs1 is p(rateR|stim1))
        # Note: conditional probs of 0 are automatically eliminated, since by convention 0*log0=0
        _,counts    = np.unique(data1, return_counts=True)
        Prs1        = counts / counts.sum()                 # P(response==r | stimulus1)
        _,counts    = np.unique(data2, return_counts=True)
        Prs2        = counts / counts.sum()                 # P(response==r | stimulus2)
                
        # Conditional entropies for stimuli 1 & 2 (Dayan & Abbott, eqn. 4.5)
        Hs1     = _entropy(Prs1)
        Hs2     = _entropy(Prs2)
    
        # Noise entropy (Dayan & Abbott, eqn. 4.6)
        Hnoise  = Ps1*Hs1 + Ps2*Hs2

        # Mutual information is total response entropy - noise entropy (Dayan & Abbott, eqn. 4.7)
        info[i_series] = H - Hnoise

    info = _unreshape_data(info,data1_shape,axis=axis)
        
    return info
  
  
def _entropy(P):
    """ Computes entropy from probabilty density P """
    return -(P * np.log2(P)).sum()  
  
  
# =============================================================================
# Area under ROC curve (AUROC) analysis
# =============================================================================
def auroc(labels, data, axis=0, signed=True, groups=None):
    """
    Mass-univariate area-under-ROC-curve metric of discriminability
    between two data distributions
    
    roc = auroc(labels,data,axis=0,signed=True,groups=None)

    Calculates area under receiver operating characteristic curve (AUROC)
    relating hits to false alarms in a binary classification/discrimination

    AUROC = 0.5 indicates no difference between distributions, and 
    AUROC = 1 indicates data distributions are completely discriminable.

    Computation is performed over trial/observation <axis>, in mass-univariate 
    fashion across data series in all other data dimensions (channels, time points, 
    frequencies, etc.).

    Note: This is a wrapper around auroc_2groups(), which accepts
    two data distributions as arguments, and is faster.

    ARGS
    labels      (n_obs,) array-like. List of group labels labelling observations from 
                each group. Should be only two groups represented, unless sub-selecting 
                two groups using <groups> argument.

    data        (...,n_obs,...) ndarray. Data values for both distributions to be compared.  

    axis        Scalar. Axis of data array to perform analysis on, corresponding
                to trials/observations. Default: first non-singleton dimension

    signed      Bool. If True [default], returns signed AUROC. If False, returns absolute
                value of AUROC, corresponding to finding "preferred" distribution for each
                data series (non-trial dims). 
                
                NOTE: absolute AUROC is a biased metric--its expected value is > 0 even 
                for identical data distributions. You might want to use resampling
                methods estimate the bias and correct for it.

    groups      (2,) array-like. Which group labels from <labels> to use. 
                Useful for enforcing a given order to groups (eg sign to signed AUROC), 
                or for computing info btwn pairs of groups in data with > 2 groups,
                Default: unique(labels) (all distinct values in <labels>)

    RETURNS
    roc         (...,1,...) ndarray.  AUROC btwn. groups along given axis. 
    """
    labels = np.asarray(labels)
    
    # Find set of unique group labels in <labels>
    if groups is None: groups = np.unique(labels)

    assert len(groups) == 2, \
        "auroc: computation only supported for 2 groups (%d given)" % len(groups)

    # Split data into 2 groups and use auroc_2groups() to actually do computation
    return auroc_2groups(data.compress(labels == groups[0], axis=axis),
                         data.compress(labels == groups[1], axis=axis), 
                         axis=axis, signed=signed)


def auroc_2groups(data1, data2, axis=0, signed=True):
    """
    Mass-univariate area-under-ROC-curve metric of discriminability
    between two data distributions
        
    roc_area = auroc_2groups(data1,data2,axis=0,signed=True)

    Calculates area under receiver operating characteristic curve (AUROC)
    relating hits to false alarms in a binary classification/discrimination

    AUROC = 0.5 indicates no difference between distributions, and 
    AUROC = 1 indicates data distributions are completely discriminable.
    
    Computation is performed over trial/observation <axis>, in mass-univariate 
    fashion across data series in all other data dimensions (channels, time points, 
    frequencies, etc.).
    
    Note: For convenience, auroc() is also provided as a wrapper around this
    function that accepts standard data,label arguments.
    
    ARGS
    data1/2     (...,n_obs1/n_obs2,...) ndarrays of arbitary size except for <axis>. 
                Sets of values for each of two distributions to be compared.  

    axis        Scalar. Dimension of data1,2 arrays to perform analysis on, corresponding
                to trials/observations. Default: first non-singleton dimension

    OPTIONAL INPUT  Additional optional parameters, given as name/value pairs in varargin:
    signed      Bool. If True [default], returns signed AUROC using given order of data1/2,
                giving a measure which can be < 0.5 if data2 is "preferred". 
                If false, rectifies AUROC around 0.5: ROCunsigned = abs(ROC-0.5)+0.5,
                essentially giving an unsigned/absolute measure of neural information 
                (ie, returning the same value whether data1 > data2 or vice versa).
        
                NOTE: unsigned AUROC is a biased metric--its expected value is > 0 even 
                for identical data distributions. You might want to use resampling
                methods estimate the bias and correct for it.

    RETURNS
    roc_area    (...,1,...) ndarray.  AUROC btwn. data1 and data2 along given axis. 

    """
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]
    
    assert (n1 != 0) and (n2 != 0), \
        "auroc: Data contains no observations (trials) for one or more groups"

    # Reshape data arrays -> (n_observations,n_data_series) matrix
    data1, data1_shape = _reshape_data(data1,axis)
    data2, data2_shape = _reshape_data(data2,axis)

    n1 = data1.shape[0]
    n2 = data2.shape[0]
    
    if data1.ndim > 1:
        n_series = data1.shape[1]
                    
        assert data1.shape[1] == data2.shape[1], \
            "auroc: Data distributions to compare must have same number of data series (timepts,freqs,channels,etc.) \
             (data1 ~ %d, data2 ~ %d)" % (data1.shape[1],data2.shape[1])
    else:
        n_series = 1
        
    roc_area = np.empty((1,n_series))        
  
    for i_series in range(n_series):    
        data1_series = data1[:,i_series] if n_series > 1 else data1.squeeze()
        data2_series = data2[:,i_series] if n_series > 1 else data2.squeeze()

        # Criterion values are all unique values in full set of both distributions
        criteria    = np.flip(np.unique(np.hstack((data1_series, data2_series))))
        n_crit      = len(criteria)

        # Calculate hit and false alarm rates for each criterion value
        # False alarm rate = proportion of data2 values "incorrectly" > each tested criterion value
        # Hit rate = proportion of data1 values "correctly" > each tested criterion value
        # For each, concatenate in the trivial end point where FA = hit rate = 1
        fa_rate  	= np.ones((n_crit+1,))
        hit_rate    = np.ones((n_crit+1,))
        for i_crit,crit in enumerate(criteria):
            fa_rate[i_crit]     = (data2_series > crit).sum()
            hit_rate[i_crit]    = (data1_series > crit).sum()
        
        fa_rate[0:-1]   = fa_rate[0:-1] / n2
        hit_rate[0:-1]  = hit_rate[0:-1] / n1

        # Calculate area under ROC curve (FA rate x hit_rate) by discrete integration
        roc_area[i_series] = (np.diff(fa_rate) * (hit_rate[:-1] + hit_rate[1:])/2).sum()
  
    # If desired, calculate absolute/unsigned area under ROC metric
    if not signed: roc_area = np.abs(roc_area - 0.5) + 0.5
  
    roc_area = _unreshape_data(roc_area,data1_shape,axis=axis)
        
    return roc_area
  

# =============================================================================
# d-prime (Cohen's d) analysis
# =============================================================================
def dprime(labels, data, axis=0, signed=True, groups=None):
    """
    Mass-univariate d' metric of difference between two data distributions

    d = dprime(labels,data,axis=0,signed=True,groups=None)

    Calculates d' (aka Cohen's d) metric of difference btwn two distributions, 
    under (weak-ish) assumption they are IID normal, using formula:
    d' = (mu1 - mu2) / sd_pooled

    d' = 0 indicates no difference between distribution means. d' is unbounded, and
    increases monotonically with the difference btwn group means and inversely with
    the pooled std deviation.

    Computation is performed over trial/observation <axis>, in mass-univariate 
    fashion across data series in all other data dimensions (channels, time points, 
    frequencies, etc.).

    Note: This is a wrapper around dprime_2groups(), which accepts
    two data distributions as arguments, and is faster.

    ARGS
    labels      (n_obs,) array-like. List of group labels labelling observations from 
                each group. Should be only two groups represented, unless sub-selecting 
                two groups using <groups> argument.

    data        (...,n_obs,...) ndarray. Data values for both distributions to be compared.  

    axis        Scalar. Axis of data array to perform analysis on, corresponding
                to trials/observations. Default: first non-singleton dimension

    signed      Bool. If True [default], returns signed d'. If False, returns absolute
                value of d', corresponding to finding "preferred" distribution for each
                data series (non-trial dims). 
                
                NOTE: absolute d' is a biased metric--its expected value is > 0 even 
                for identical data distributions. You might want to use resampling
                methods estimate the bias and correct for it.

    groups      (2,) array-like. Which group labels from <labels> to use. 
                Useful for enforcing a given order to groups (sign to results d'), 
                or for computing d' btwn pairs of groups in data with > 2 groups,
                Default: unique(labels) (all distinct values in <labels>)

    RETURNS
    d           (...,1,...) ndarray.  d' btwn. groups along given axis. 

    REFERENCE   Dayan & Abbott _Theoretical Neuroscience_ eqn. 3.4 (p.91)
    """
    labels = np.asarray(labels)
    
    # Find set of unique group labels in <labels>
    if groups is None: groups = np.unique(labels)

    assert len(groups) == 2, \
        "dprime: d' computation only supported for 2 groups (%d given)" % len(groups)

    # Split data into 2 groups and use dprime_2groups() to actually do computation
    return dprime_2groups(data.compress(labels == groups[0], axis=axis),
                          data.compress(labels == groups[1], axis=axis), 
                          axis=axis, signed=signed)


def dprime_2groups(data1, data2, axis=0, signed=True):
    """
    Mass-univariate d' metric of difference between two data distributions
    
    d = dprime_2groups(data1,data2,axis=0,signed=True)

    Calculates d' (aka Cohen's d) metric of difference btwn two distributions, 
    under (weak-ish) assumption they are IID normal, using formula:
    d' = (mu1 - mu2) / sd_pooled
    
    d' = 0 indicates no difference between distribution means. d' is unbounded, and
    increases monotonically with the difference btwn group means and inversely with
    the pooled std deviation.
    
    Computation is performed over trial/observation <axis>, in mass-univariate 
    fashion across data series in all other data dimensions (channels, time points, 
    frequencies, etc.).
    
    Note: For convenience, dprime() is also provided as a wrapper around this
    function that accepts standard data,label arguments.
    
    ARGS
    data1/2     (...,n_obs1/n_obs2,...) ndarrays of arbitary size except for <axis>. 
                Sets of values for each of two distributions to be compared.  

    axis        Scalar. Dimension of data1,2 arrays to perform analysis on, corresponding
                to trials/observations. Default: first non-singleton dimension

    OPTIONAL INPUT  Additional optional parameters, given as name/value pairs in varargin:
    signed      Bool. If True [default], returns signed d'. If False, returns absolute
                value of d', corresponding to finding "preferred" distribution for each
                data series (non-trial dims).

                NOTE: absolute d' is a biased metric--its expected value is > 0 even 
                for identical data distributions. You might want to use resampling
                methods estimate the bias and correct for it.

    RETURNS
    d           (...,1,...) ndarray.  d' btwn. data1 and data2 along given axis. 

    REFERENCE   Dayan & Abbott _Theoretical Neuroscience_ eqn. 3.4 (p.91)
    """
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]
    
    assert (n1 != 0) and (n2 != 0), \
        "dprime: Data contains no observations (trials) for one or more groups"

    # Compute difference of group means
    d = data1.mean(axis=axis,keepdims=True) - data2.mean(axis=axis,keepdims=True)

    # Compute group std dev's
    sd1	= data1.std(axis=axis,ddof=1,keepdims=True)
    sd2	= data2.std(axis=axis,ddof=1,keepdims=True)
    # Compute pooled standard deviation across two groups, using standard formula
    sdPooled = np.sqrt( ((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2) )

    d = d / sdPooled

    # Deal w/ special case of mu1-mu2 = 0 and both SD's = 0 
    # (identical distributions for 2 cond's) -> d' = 0
    d[np.isnan(d)] = 0

    # Return unsigned (absolute) d', if requested
    if not signed: d = np.abs(d)

    return d
  
  
# =============================================================================
# Percent explained variance (PEV) and specific linear model functions for PEV analysis
# =============================================================================
def pev(X, y, axis=0, model=None, as_pct=True, return_stats=False, **kwargs):
    """
    Mass-univariate percent explained variance (PEV) analysis.

    Computes the percentage (or proportion) of variance explained in data y by
    predictors in design matrix X, using one of a few types of linear models.

    exp_var = pev(X,y,axis=0,model=None,as_pct=True,return_stats=False,**kwargs)
    exp_var,stats = pev(X,y,axis=0,model=None,as_pct=True,return_stats=False,**kwargs)

    ARGS
    X       (n_obs,n_terms) array-like. Design matrix (group labels for ANOVA models,
            or regressors for regression model) for each observation (trial).
            X.shape[0] must be same length as observation <axis> of <y>.

    y       (...,n_obs,...) ndarray. Data to fit with linear model. Axis <axis> should
            correspond to observations (trials), while any other axes can be any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels <X>.

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    model   String. Type of linear model to fit, in order to compute PEV.
            'anova1'    : 1-way ANOVA model. X is (n_obs,) vector)
            'anova2'    : 2-way ANOVA model (X must be a (n_obs,2) array)
            'anovan'    : n-way ANOVA model (X must be a (n_obs,n_terms) array)
            'regress'   : linear regression model (X is (n_obs,nModelParams) array)
            Default: we attempt to infer from <X>. Safest to set explicitly.

    as_pct  Bool. Set=True [default] to return PEV as a percent (range ~ 0-100).
            Otherwise PEV returned as a proportion (range ~ 0-1)

    return_stats Bool. Set=True to return several stats on model fit (eg F-stat,p)
            in addition to PEV. Otherwise, just returns PEV. Default: False

    **kwargs Passed directly to model function. See those for details.

    RETURNS
    exp_var (...,n_terms,...). Percent (or proportion) of variance in y explained by X
            Shape is same as y, with observation axis reduced to length = n_terms.

    stats   Dict. If <return_stats> set, statistics on each fit also returned.
            See model function for specific stats returned by each.
    """
    # TODO Add anovan model
    X = np.asarray(X)

    # Attempt to infer proper linear model based on X
    if model is None:
        # If X is vector-valued, assume 1-way ANOVA model
        if (X.ndim == 1) or (X.shape[1] == 1):  model = 'anova1'
        # If X has constant/intercept term (column of all 1's), assume regression model
        elif (X == 1).all(axis=0).any():        model = 'regress'
        # If X has > 3 columns, assume n-way ANOVA
        # TODO ADD: elif X.shape[1] > 3:                    model = 'anovan'
        # Otherwise, could be ANOVA2, ANOVAn, regress ... dangerous to assume
        else:
            raise ValueError("Could not determine appropriate linear model.\n" \
                             "Please set explicitly using <model> argument.")
                
        print("Assuming '%s' linear model based on given <X> labels/design matrix" % model)
        
    model = model.lower()
    
    # Compute PEV based on 1-way ANOVA model
    if model == 'anova1':
        return anova1(X,y,axis=axis,as_pct=as_pct,return_stats=return_stats,**kwargs)
    # Compute PEV based on 2-way ANOVA model
    elif model == 'anova2':
        return anova2(X,y,axis=axis,as_pct=as_pct,return_stats=return_stats,**kwargs)
    # Compute PEV based on 2-way ANOVA model
    elif model == 'regress':
        return regress(X,y,axis=axis,as_pct=as_pct,return_stats=return_stats,**kwargs)
    else:
        raise ValueError("'%s' model is not supported for computing PEV" % model)

percent_explained_variance = pev
""" Aliases function pev as percent_explained_variance """


def anova1(X, y, axis=0, omega=True, gm_method='mean_of_obs',
           as_pct=True, return_stats=False):
    """
    Mass-univariate 1-way ANOVA analyses of one or more data vector(s) y
    on single list of group labels X

    exp_var = anova1(X,y,axis=0,omega=True,gm_method='mean_of_obs',
                     as_pct=True,return_stats=False)

    exp_var,stats = anova1(X,y,axis=0,omega=True,gm_method='mean_of_obs',
                           as_pct=True,return_stats=False)

    ARGS
    X       (n_obs,) array-like. Group labels for each observation (trial),
            identifying which group/factor level each observation belongs to.
            Number of rows(X.shape[0] = n_obs) be same length as y.shape[axis].

    y       (...,n_obs,...) ndarray. Data to fit with ANOVA model. <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels X.

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    omega   Bool. Determines formula for calculating PEV.  Default: True
            True  : Bias-corrected omega-squared PEV formula [default]
            False : Standard eta-squared formula. Positively biased for small N.

    gm_method String. Method used to calculate grand mean for ANOVA formulas.
            'mean_of_obs'   : Mean of all observations (more standard ANOVA formula)
            'mean_of_means' : Mean of group means--less downward-biasing of PEV,F
                              for unbalanced grp n's
            Default: 'mean_of_obs'

    as_pct  Bool. Set=True [default] to return PEV as a percent (range ~ 0-100).
            Otherwise PEV returned as a proportion (range ~ 0-1)

    return_stats Bool. Set=True to return several stats on model fit (eg F-stat,p)
            in addition to PEV. Otherwise, just returns PEV. Default: False

    RETURNS
    exp_var (..,1,...). Percent (or proportion) of variance in y explained by X.
            Shape is same as y, with observation axis reduced to length 1.

    stats   Dict. If <return_stats> set, statistics on each fit also returned:
        p   (...,1,...). F-test p values for each datapoint. Same shape as exp_var.
        F   (...,1,...). F-statistic for each datapoint. Same shape as exp_var.
        mu  (...,n_groups,...). Group mean for each group/level
        n   (...,n_groups,). Number of observations (trials) in each group/level
    """
    # TODO Add <groups> arg with list of group labels to use
    # Reshape data array y -> (n_observations,n_data_series) matrix
    y, y_shape = _reshape_data(y,axis)
    n_obs,n_series = y.shape

    assert (X.ndim == 1) or (X.shape[1] == 1), \
            "Design matrix should have only a single column for anova1 model (it has %d)" \
            % X.shape[1]
    assert X.shape[0] == n_obs, \
            "Design matrix X and data array y should have same number of rows (%d != %d)" \
            % (X.shape[0], n_obs)

    # Find set of groups (levels) in list of labels
    groups = np.unique(X)
    n_groups = len(groups)

    # Compute mean for each group (and for each data series)
    n = np.empty((n_groups,))
    mu = np.empty((n_groups,n_series))

    for i_group,group in enumerate(groups):
        group_idxs   = X == group
        n[i_group]   = group_idxs.sum()   # Number of observations for given group
        mu[i_group,:]= y[group_idxs,:].mean(axis=0)  # Group mean for given group

    # Compute grand mean across all observations (for each data series)
    if gm_method == 'mean_of_obs':        grand_mean = y.mean(axis=0)
    elif gm_method == 'mean_of_means':    grand_mean = mu.mean(axis=0)

    # Total Sums of Squares
    SS_total = ((y - grand_mean)**2).sum(axis=0)

    # Groups Sums of Squares
    SS_groups = np.zeros((n_series,))
    for i_group in range(n_groups):
        # Group Sum of Squares for given group
        SS_groups += n[i_group]*(mu[i_group,:] - grand_mean)**2

    # Compute degrees of freedom and Mean Squares needed for PEV and/or F-stat formulas
    if omega or return_stats:
        df_groups= n_groups - 1         # Groups degrees of freedom
        df_error = n_obs-1 - df_groups  # Error degrees of freedom
        MS_error = (SS_total - SS_groups) / df_error   # Error mean square

    # PEV,F strictly undefined when no y variance (div by 0).
    # By convention, set = 0 (below) for these cases.
    undefined = SS_total == 0
    SS_total[undefined] = 1     # Set=1 to avoid annoying divide-by-0 warnings

    # Omega-squared stat = bias-corrected explained variance
    if omega:   exp_var = omega_squared(SS_groups,SS_total,MS_error,df_groups)
    # Standard eta-squared formula
    else:       exp_var = eta_squared(SS_groups,SS_total)

    exp_var[:,undefined] = 0

    # Convert proportion [0-1] -> percent [0-100]
    if as_pct:   exp_var = 100.0*exp_var

    exp_var = _unreshape_data(exp_var,y_shape,axis=axis)

    if not return_stats:
        return exp_var

    # Calculate F-statistic and perform F-test to determine p value for all data points
    else:
        MS_groups= SS_groups / df_groups    # Groups mean square
        F       = MS_groups / MS_error      # F statistic
        F[:,undefined] = 0                  # Set F = 0 for data w/ y variance = 0
        p       = Ftest.sf(F,df_groups,df_error) # p value for given F stat

        F       = _unreshape_data(F,y_shape,axis=axis)
        p       = _unreshape_data(p,y_shape,axis=axis)
        mu      = _unreshape_data(mu,y_shape,axis=axis)

        stats   = {'p':p, 'F':F, 'mu':mu, 'n':n}
        return exp_var, stats


def anova2(X, y, axis=0, interact=None, omega=True, partial=False, total=False,
           gm_method='mean_of_obs', as_pct=True, return_stats=False):
    """
    Mass-univariate 2-way ANOVA analyses of one or more data vector(s) y
    on single set of group labels X

    exp_var = anova2(X,y,axis=0,interact=None,omega=True,partial=False,total=False,
                     gm_method='mean_of_obs',as_pct=True,return_stats=False)

    exp_var,stats = anova2(X,y,axis=0,interact=None,omega=True,partial=False,total=False,
                           gm_method='mean_of_obs',as_pct=True,return_stats=False)

    ARGS
    X       (n_obs,n_terms=2|3) array-like. Group labels for each observation (trial),
            identifying which group/factor level each observation belongs to.
            Can either set interaction term labels for column 3, or set
            interact==True and we will auto-generate interaction term
            Number of rows(X.shape[0] = n_obs) be same length as y.shape[axis].

    y       (...,n_obs,...) ndarray. Data to fit with ANOVA model. <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels X.

    axis    Int. Data axis corresponding to distinct observations/trials. Default: 0

    interact Bool. Determines whether an interaction term is included in model.
            If set, but but no 3rd entry is given in X, we auto-generate an
            interaction term based on all unique combinations of X1 & X2 levels.
            Default: true iff X has 3rd column/entry (for interaction)

    omega   Bool. Determines formula for calculating PEV.  Default: True
              True  : Bias-corrected omega-squared PEV formula [default]
              False : Standard eta-squared formula. Positively biased for small N.

    partial Logical. Determines method used to calc PEV -- full-model or partial.
              False : Standard full-model PEV = SS_factor / SS_total.
                      Increase in PEV for one factor will decrease all others.
              True  : Partial factor PEV = SS_factor / (SS_factor + SS_error).
                      Factor EV's are therefore independent of each other.
            Default: False

    total   Bool. Set=True to append total PEV, summed across all model terms,
            to end of terms axis in <exp_var>.  Default: False

    as_pct   Bool. Set=True [default] to return PEV as a percent (range ~ 0-100).
            Otherwise PEV returned as a proportion (range ~ 0-1)

    gm_method String. Method used to calculate grand mean for ANOVA formulas.
            'mean_of_obs'   : Mean of all observations (more standard ANOVA formula)
            'mean_of_means' : Mean of group (cell) means--less downward-biasing of PEV,F
                            for unbalanced grp n's. For this option, MUST have interact==True.
            Default: 'mean_of_obs'

    return_stats Bool. Set=True to return several stats on model fit (eg F-stat,p)
            in addition to PEV. Otherwise, just returns PEV. Default: False

    RETURNS
    exp_var (n_terms,...). Percent (or proportion) of variance in y explained by X.
            Shape is same as y, with observation axis reduced to length = n_terms.

    stats   Dict. If <return_stats> set, statistics on each fit also returned:
        p   (...,n_terms,...). F-test p values for each datapoint. Same shape as exp_var.
        F   (...,n_terms,...). F-statistic for each datapoint. Same shape as exp_var.
        mu  [n_terms] list of (...,n_groups,...). Group mean for each group (level),
            in a separate list element for each model term (b/c n_groups not same).
        n   [n_terms] list of (...,n_groups,...). Number of observations (trials)
            in each group/level, in a separate list element for each model term.

    REFERENCE   Zar _Biostatistical Analysis_ 4th ed.
    """
    # TODO Add <groups> arg with list of group labels to use
    # Reshape data array y -> (n_observation,n_data_series) matrix
    y, y_shape  = _reshape_data(y,axis)
    n_obs,n_series= y.shape

    # If interaction term is given in X, its an interaction model
    if interact is None:  interact = X.shape[1] == 3
    n_terms      = 2 + interact

    assert X.shape[0] == n_obs, \
            "Design matrix X and data array y should have same number of rows (%d != %d)" \
            % (X.shape[0], n_obs)
    assert X.shape[1] in [2,3], \
            "Design matrix should have 2 or 3 columns for anova2 model (it has %d)" \
            % X.shape[1]

    # If interaction term is requested, but not provided, create one here
    if interact and X.shape[1] == 2:
        X = np.concatenate((X, np.zeros((n_obs,),dtype=int)))
        group_pairs = np.unique(X[:,0:2])   # All unique combinations of factor 1 & 2
        for i_pair,pair in enumerate(group_pairs):
            idxs = np.all(X == pair, axis=1)# Find and label all observations (trials)
            X[idxs,2] = i_pair              # with given pair of groups/levels

    groups = [np.unique(X[:,i_term]) for i_term in range(n_terms)]
    n_groups = np.asarray([len(term_groups) for term_groups in groups])

    # Compute means for each group (factor level) in dataset
    n  = []
    mu = []
    for i_term in range(n_terms):
        n.append(np.zeros((n_groups[i_term],)))
        mu.append(np.zeros((n_groups[i_term],n_series)))

        for i_group,group in enumerate(groups[i_term]):
            group_idxs = X[:,i_term] == group           # Find all obs for current group
            n[i_term][i_group]    = np.sum(group_idxs)  # n for given group
            # Group means for given group
            mu[i_term][i_group,:] = np.mean(y[group_idxs,:], axis=0)

    # Compute grand mean across all observations (for each data series)
    if gm_method == 'mean_of_obs':
        grand_mean = np.mean(y, axis=0)         # Grand mean of all data points
    else: # implicit: gm_method == 'mean_of_means'
        grand_mean = np.mean(mu[2], axis=0)     # Grand mean of cell means

    # Compute groups (effects) Sums of Squares for all data points for each term
    SS_groups = np.zeros((n_terms,n_series))
    for i_term in range(n_terms):
        for i_group,group in enumerate(groups[i_term]):
            # Group Sum of Squares for given group
            SS_groups[i_term,:] += n[i_term][i_group]*(mu[i_term][i_group,:] - grand_mean)**2

        # For interaction term, calculations above give Cells Sum of Squares (Zar eqn. 12.18).
        # Remove main effects sums of squarea to get interaction (Zar eqn. 12.12).
        if i_term == 2:
            SS_groups[i_term,:] -= (SS_groups[0,:] + SS_groups[1,:])

    SS_total = np.sum((y - grand_mean)**2, axis=0)  # Total Sum of Squares
    SS_error = SS_total - np.sum(SS_groups, axis=0) # Error (residual, within-cells) Sum of Squares

    # Compute degrees of freedom and Mean Squares needed for PEV and/or F-stat formulas
    if omega or return_stats:
        df_total    = n_obs - 1             # Total degrees of freedom
        df_groups   = n_groups - 1          # Groups degrees of freedom (Zar eqn. 12.9)
        df_cells    = df_groups[-1]         # Cells degrees of freedom (Zar eqn. 12.4)
        if interact:
            # Interaction term degrees of freedom = df_cells - dfMain1 - dfMain2 (Zar eqn. 12.13)
            df_groups[2] -= (df_groups[0] + df_groups[1])
        df_error    = df_total - df_cells   # Error degrees of freedom (Zar eqn. 12.7)
        MS_error    = SS_error / df_error   # Error Mean Squares
        if axis != -1:
            df_groups= df_groups.reshape((*np.ones((axis,),dtype=int),
                                          n_terms,
                                          *np.ones((SS_groups.ndim-axis-1,),dtype=int)))

    # PEV,F strictly undefined when no y variance (div by 0).
    # By convention, set = 0 (below) for these cases.
    undefined = SS_total == 0
    SS_total[undefined] = 1     # Set=1 to avoid annoying divide-by-0 warnings

    # Calculate explained variance
    if not partial:     # Standard (full-model) PEV
        if omega: exp_var = omega_squared(SS_groups,SS_total,MS_error,df_groups)
        else:     exp_var = eta_squared(SS_groups,SS_total)

    else:               # Partial factor PEV
        if omega: exp_var = omega_squared_partial(SS_groups,SS_total,MS_error,df_groups,n_obs)
        else:     exp_var = eta_squared_partial(SS_groups,SS_error)

    exp_var[:,undefined] = 0

    if as_pct:   exp_var = 100.0*exp_var     # Convert proportion [0-1] -> percent [0-100]

    exp_var = _unreshape_data(exp_var,y_shape,axis=axis)

    # Append summed PEV across all model terms to end of term axis
    if total:   exp_var = np.concatenate((exp_var, np.sum(exp_var,axis=0,keepdims=True)), axis=0)

    if not return_stats:
        return exp_var

    # Calculate F-statistic and perform F-test to determine p value for all data points
    else:
        MS_groups= SS_groups/df_groups  # Groups Mean Squares for each term
        F       = MS_groups/MS_error    # F statistics for each term
        F[:,undefined] = 0              # Set F = 0 for data w/ y variance = 0
        p       = Ftest.sf(F,df_groups,df_error)    # p value for given F stat

        F       = _unreshape_data(F,y_shape,axis=axis)
        p       = _unreshape_data(p,y_shape,axis=axis)
        mu      = [_unreshape_data(mu[i_term],y_shape,axis=axis) for i_term in range(n_terms)]

        stats   = {'p':p, 'F':F, 'mu':mu, 'n':n}
        return exp_var, stats


def regress(X, y, axis=0, X_terms=None, constant=True, total=False,
            omega=True, partial=False, as_pct=True, return_stats=False):
    """
    Mass-univariate ordinary least squares regression analyses of one or more
    data vector(s) y on single design matrix X

    exp_var = regress(X,y,axis=0,X_terms=None,constant=True,total=False,
                  omega=True,partial=False,as_pct=True,return_stats=False)

    exp_var,stats = regress(X,y,axis=0,X_terms=None,constant=True,total=False,
                        omega=True,partial=False,as_pct=True,return_stats=False)

    ARGS
    X       (n_obs,n_params) array-like. Regression design matrix. Each row corresponds
            to a distinct observation (trial), and each column to a distinct
            predictor (coefficient to fit). If <constant> == True, a constant
            (intercept) column will be appended to end of X, if not already present.
            Number of rows(X.shape[0] = n_obs) be same length as y.shape[axis].

    y       (...,n_obs,...) ndarray. Data to fit with regression model. <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels X.

    axis    Int. Data axis corresponding to distinct observations/trials. Default: 0

    X_terms  (n_params,) array-like. Lists regression term (eg as integer or string
            name) corresponding to each column (predictor) in X. Mapping may not
            be 1:1 due to multiple dummy-variable columns arising from
            categorical terms with > 2 levels. PEV/stats are computed separately
            for all columns/predictors of each term pooled together.
            Default: np.arange(n_params) = 0:n_params (1:1 mapping from term:column)

    constant Bool. If True, ensures there is a constant column in X to fit an
            intercept/bias term (appends if missing, does nothing if present).
            Default: True (include constant/intercept term)

    total   Bool. If True, appends total model explained variance (sum of all
            individual terms) to end of term axis. Default: False

    omega   Bool. If True, uses bias-corrected omega-squared formula for PEV,
            otherwise uses R-squared formula, which is positively biased.
            Default: True

    partial Bool. If True, uses partial-factor formula for PEV, where each term
            EV is expressed relative to only that term + error variance, and
            thus changes in one term's EV do not necessarily affect other terms.
            Otherwise, the standard full-model PEV formula is used. Default: False

    as_pct  Bool. If True, returns PEV as a percent (range ~ 0-100), else PEV
            is returned as a proportion (range ~ 0-1). Default: True

    return_stats Bool. If True, computes and returns several statistics of fitted
            model in addition to PEV. Default: False

    RETURNS
    exp_var (...,n_terms,...). Percent (or proportion) of variance in y explained by X.
            Shape is same as y, with observation axis reduced to length = n_terms.

    stats   Dict. If <return_stats> set, statistics on each fit also returned:
        p   (...,n_terms,...). F-test p values for each datapoint. Same shape as exp_var.
        F   (...,n_terms,...). F-statistic for each datapoint. Same shape as exp_var.
        B   (...,nPredictors,...). Fitted regression coefficients for each predictor
            (column in X). Same shape as exp_var, but with B.shape[axis] = nPredictors.

    REFERENCE
    Regression eqn's: Draper & Smith _Applied Regression Analysis (1998) sxn's 1.3, 6.1
    Omega^2 stat:     Snyder & Lawson (1993) J of Experimental Education
                      wikipedia.org/wiki/Effect_size
    """
    # Reshape data array y -> (n_observations,n_data_series) matrix
    y, y_shape = _reshape_data(y,axis)
    n_obs,n_series = y.shape

    assert X.shape[0] == n_obs, \
            "Design matrix X and data array y should have same number of rows (%d != %d)" \
            % (X.shape[0], n_obs)

    # If X_terms not set, assume each column in X is a distinct term
    if X_terms is None: X_terms = np.arange(X.shape[1])
    X_terms      = np.asarray(X_terms)

    # If a constant is requested and not already present in design matrix X,
    #  concatenate (n_obs,) vector of ones to end of design matrix
    constant_col = np.all(X==1,axis=0)
    if constant and not np.any(constant_col):
        X       = np.hstack((X,np.ones((n_obs,1))))
        X_terms  = np.hstack((X_terms,np.nan))
        constant_col = np.hstack((constant_col,True))

    n_params    = X.shape[1]

    term_set    = _unsorted_unique(X_terms[~constant_col])
    n_terms     = len(term_set)

    grand_mean  = np.mean(y, axis=0)
    SS_total    = np.sum((y - grand_mean)**2, axis=0) # Total Sums of Squares

    # Create linear regression object
    model       = LinearRegression()

    # Fit full model to y, save coefficients, and compute prediction of y
    model.fit(X,y)
    # Reshape coeffs (n_series,nPredictors) -> (nPredictors,n_series)
    B           = model.coef_.T
    # Full-model Error Sums of Squares
    SS_error_full= np.sum((y - model.predict(X))**2, axis=0)

    # Fit reduced models eliminating each term, in turn, and calculate
    #  additional sums of squares for each (ie full model minus each factor)
    SS_extra = np.zeros((n_terms,n_series))
    df_extra = np.zeros((n_terms,))
    for i_term,term in enumerate(term_set):
        term_idxs= X_terms == term
        df_extra[i_term] = np.sum(term_idxs)   # Regression degrees of freedom for term

        # Fit/predict with reduced model, without current-term X columns
        X_reduced= X[:,~term_idxs]
        model.fit(X_reduced,y)
        # Reduced-model Error Sums of Squares
        SSerror_reduced = np.sum((y - model.predict(X_reduced))**2, axis=0)
        # Extra regression Sums-of-Squares for term
        SS_extra[i_term,:]= SSerror_reduced - SS_error_full

    # PEV,F strictly undefined when no y variance (div by 0).
    # By convention, set = 0 (below) for these cases.
    undefined = SS_total == 0
    # Set 0-variance entries = 1 to avoid annoying divide-by-0 warnings
    # (values are overwritten with 0's below)
    SS_total[undefined] = 1
    SS_error_full[undefined] = 1

    if omega or return_stats:
        df_reg_full  = n_params - 1   # Regression Degrees of freedom
        df_total     = n_obs - 1      # Total (corrected) Degrees of freedom
        df_error     = df_total - df_reg_full  # Error/Residual Degrees of freedom
        MS_error     = SS_error_full/df_error  # Mean Squares due to residual error

    # Calculate explained variance
    if not partial:     # Standard (full-model) PEV
        if omega:   exp_var = omega_squared(SS_extra,SS_total,MS_error,df_extra)
        else:       exp_var = R_squared(SS_extra,SS_total)

    else:               # Partial factor PEV
        if omega:   exp_var = omega_squared_partial(SS_extra,SS_total,MS_error,df_extra,n_obs)
        else:       exp_var = R_squared_partial(SS_extra,SS_error_full)

    exp_var[:,undefined] = 0

    if as_pct:   exp_var = 100.0*exp_var     # Convert proportion [0-1] -> percent [0-100]

    exp_var = _unreshape_data(exp_var,y_shape,axis=axis)

    # Append summed PEV across all model terms to end of term axis
    if total:
        exp_var = np.concatenate((exp_var, exp_var.sum(axis=0,keepdims=True)), axis=0)
        if return_stats:
            df_extra = np.concatenate((df_extra, df_extra.sum(axis=0,keepdims=True)), axis=0)
            SS_extra = np.concatenate((SS_extra, SS_extra.sum(axis=0,keepdims=True)), axis=0)

    if not return_stats:
        return exp_var

    # Calculate "extra-sums-of-squares" F-statistic and associated p value
    else:
        df_extra    = np.reshape(df_extra,(-1,1))
        MS_regress  = SS_extra / df_extra     # Regression Mean Squares for each term
        F           = MS_regress / MS_error   # F statistics for each term
        F[:,undefined] = 0                    # Set F = 0 for data w/ y variance = 0
        p           = Ftest.sf(F,df_extra,df_error) # p value for given F stat

        F       = _unreshape_data(F,y_shape,axis=axis)
        p       = _unreshape_data(p,y_shape,axis=axis)
        B       = _unreshape_data(B,y_shape,axis=axis)

        stats   = {'p':p, 'F':F, 'B':B}
        return exp_var, stats


# =============================================================================
# Design matrix-related functions
# =============================================================================
def patsy_terms_to_columns(X):
    """
    Given a patsy DesignMatrix, maps model terms to design matrix columns,
    returning a vector listing the term corresponding to each column.

    Note that this correspondence may not be 1:1 due to categorical terms
    generating multiple dummy columns.

    X_terms = patsy_terms_to_columns(X)

    ARGS
    X       (n_observations,nColumns) patsy DesignMatrix object.

    RETURNS
    X_terms  (nColumns,) array of strings. Lists term name (from X.design_info.term_names)
            corresponing to each column in design matrix X.
    """
    # todo  Should we option list of int indexes instead string names? faster downstream?
    assert isinstance(X,DesignMatrix), \
        ValueError("patsy_terms_to_columns: X must be a patsy DesignMatrix object")

    n_cols  = X.shape[1]
    X_terms = np.full((n_cols,),fill_value='',dtype=object)

    # For each term in design matrix, find all columns it maps to from design_info
    # attribute in patsy DesignMatrix object, and insert term name string
    # into corresponding columns in output variable
    for term,slicer in X.design_info.term_name_slices.items():
        X_terms[slicer] = term

    return X_terms


# =============================================================================
# Low-level PEV computation functions
# =============================================================================
def R_squared(SS_model, SS_total):
    """
    Computes full-model R-squared/eta-squared statistic of explained variance.
    Statistic is positively biased, especially for small N.
    Formula :   exp_var = SS_model / SS_total
    Also aliased as eta_squared()

    exp_var = R_squared(SS_model,SS_total)
    """
    return SS_model / SS_total

eta_squared = R_squared   # alias for Rsquared -- same formula


def R_squared_partial(SS_model, SS_error):
    """
    Computes partial R-squared/eta-squared statistic of explained variance.
    Statistic is positively biased, especially for small N.
    Formula :   pev = SS_model / (SS_model + SS_error)
    Also aliased as eta_squared_partial()

    pev = R_squared_partial(SS_model,SS_error)
    """
    return SS_model / (SS_model + SS_error)

eta_squared_partial = R_squared_partial     # alias for Rsquared -- same formula


def omega_squared(SS_model, SS_total, MS_error, df_model):
    """
    Computes full-model omega-squared statistic of explained variance.
    Statistic is bias-corrected, unlike R-squared/eta-squared.
    Formula :   pev = (SS_model - df_model*MS_error) / (SS_total + MS_error)

    pev = omega_squared(SS_model,SS_total,MS_error)

    REFERENCE
        Olejnik & Algina (2003) Psychological Methods
        Snyder & Lawson (1993) J of Experimental Education
    """
    return (SS_model - np.outer(df_model,MS_error)) / (SS_total + MS_error)


def omega_squared_partial(SS_model,SS_total,MS_error,df_model,n_obs):
    """
    Computes partial omega-squared statistic of explained variance.
    Statistic is bias-corrected, unlike R-squared/eta-squared.
    Formula :   pev = (SS_model - df_model*MS_error) / (SS_total + (n_obs-df_model)*MS_error)

    pev = omega_squared_partial(SS_model,SS_total,MS_error,n_obs)

    REFERENCE
        Olejnik & Algina (2003) Psychological Methods
        Snyder & Lawson (1993) J of Experimental Education
    """
    return ((SS_model - np.outer(df_model,MS_error)) /
            (SS_total + np.outer((n_obs-df_model),MS_error)))



# =============================================================================
# Data reshaping helper functions
# =============================================================================
def _reshape_data(data, axis=0):
    """
    Reshapes multi-dimensional data array to 2D array form for analysis

    data, data_shape = _reshape_data(data,axis=0)

    ARGS
    data    (...,n_obs,...) ndarray. Data array where <axis> is observations (trials),
            and rest of axes (if any) represent independent data series.

    axis    Int. Axis of data coresponding to observations/trials, which is
            to be moved to axis 0 for subsequent analysis. Default: 0

    RETURNS
    data    (n_obs,n_series) ndarray. Data array with <axis> moved to axis=0, and
            all other axes unwrapped into single axis, where n_series = prod(shape[1:])

    data_shape Tuple. Original shape of data array

    Note:   Even 1d (vector) data is expanded into 2d (n_obs,1) array to
            standardize for calling code.
    """
    data = np.asarray(data)
    data_shape = data.shape         # Shape of original data
    data_ndim  = len(data_shape)    # Number of dimensions in original data


    if ~data.flags.c_contiguous:
        # If observation axis != 0, permute axis to make it so
        if axis != 0:       data = np.moveaxis(data,axis,0)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix
        if data_ndim > 2:   data = np.reshape(data,(data_shape[axis],-1),order='F')

    # Faster method for c-contiguous arrays
    else:
        # If observation axis != last dim, permute axis to make it so
        lastdim = data_ndim - 1
        if axis != lastdim: data = np.moveaxis(data,axis,lastdim)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix
        if data_ndim > 2:   data = np.reshape(data,(-1,data_shape[axis]),order='C').T

    # Expand (n_obs,) vector data to (n_obs,1) to simplify downstream code
    if data_ndim == 1:  data = data[:,np.newaxis]

    return data, data_shape


def _unreshape_data(data, data_shape, axis=0):
    """
    Reshapes data array from unwrapped 2D (matrix) form back to ~ original
    multi-dimensional form

    data = _unreshape_data(data,data_shape,axis=0)

    ARGS
    data    (axis_len,n_series) ndarray. Data array w/ all axes > 0 unwrapped into
            single dimension, where n_series = prod(shape[1:])

    data_shape Tuple. Original shape of data array

    axis    Int. Axis of original data corresponding to distinct observations,
            which has become axis 0, but will be permuted back to original axis.
            Default: 0

    RETURNS
    data       (axis_len,...) ndarray. Data array reshaped back to original shape
    """
    # Wrap negative axis back into 0 to ndim-1
    axis_ = len(data_shape) + axis if axis < 0 else axis
    data_shape  = np.asarray(data_shape)

    data_ndim = len(data_shape) # Number of dimensions in original data
    axis_len  = data.shape[0]   # Length of dim 0 (will become dim <axis> again)

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    # (but with length of dimension <axis> = <axisLength>)
    if data_ndim > 2:
        # Reshape data -> (axis_len,<original shape w/o <axis>>)
        shape = (axis_len,*data_shape[np.arange(data_ndim) != axis_])
        data = np.reshape(data,shape,order='F')

    # Squeeze (n_obs,1) array back down to 1d (n_obs,) vector
    elif data_ndim == 1:  data = np.squeeze(data)

    # If observation axis wasn't 0, permute axis back to original position
    if axis_ != 0: data = np.moveaxis(data,0,axis_)

    return data


def _unsorted_unique(x):
    """
    Implements np.unique(x) without sorting, ie maintains original order of unique
    elements as they are found in x.

    SOURCE  stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    """
    x    = np.asarray(x)
    idxs = np.unique(x,return_index=True)[1]
    return x[np.sort(idxs)]


#==============================================================================
# Testing functions
#==============================================================================
def test_info(method, test='gain', values=None, plot=False, n_reps=100, **kwargs):
    """
    Basic testing for functions estimating neural information.
    
    Generates synthetic spike rate data, estimates information using given method,
    and compares estimated to expected values.
    
    info,sd = test_info(method,test='gain',plot=False,n_reps=20, **kwargs)
                              
    INPUTS
    method  String. Name of information function to test:
            'pev' | 'anova1' | 'anova2' | 'regress'
            
    test    String. Type of test to run. Default: 'gain'. Options:
            'gain'  Tests multiple values between-condition rate difference (gain)
                    Checks for monotonically increasing information
            'n'     Tests multiple values of number of trials (n)
                    Checks that information doesn't vary with n.                    
            'bias'  Tests multiple n values with 0 btwn-cond difference
                    Checks that information is not > 0 (unbiased)
            'n_conds' Tests multiple values for number of conditions
                    (no actual checking, just to see behavior of info measure)
            
    values  (n_values,) array-like. List of values to test. 
            Interpretation and defaults are test-specific:
            'gain'  Btwn-condition rate differences (gains). Default: [1,2,5,10,20]
            'n'     Trial numbers. Default: [25,50,100,200,400,800]
            
    plot    Bool. Set=True to plot test results. Default: False
    
    n_reps  Int. Number of independent repetitions of tests to run. Default: 100
    
    **kwargs All other keyword args passed to information estimation function
    
    RETURNS
    info    (n_values,) ndarray. Estimated information for each tested value
    sd      (n_values,) ndarray. Across-run SD of information for each tested value
    
    ACTION
    Throws an error if any estimated information is too far from expected value
    If <plot> is True, also generates a plot summarizing estimated information
    """    
    from spike_analysis import simulate_spike_rates
    
    test = test.lower()
    method = method.lower()
    
    # Set defaults for tested values and set up rate generator function depending on <test>
    if test == 'gain':
        values = [1,2,5,10,20] if values is None else values
        gen_rates = lambda gain,seed: simulate_spike_rates(gain=float(gain),offset=5.0,
                                                           n_conds=2,n_trials=1000,
                                                           seed=seed)
        
    elif test in ['n','n_trials','bias']:
        values = [25,50,100,200,400,800] if values is None else values
        gain = 0.0 if test == 'bias' else 5.0
        gen_rates = lambda n_trials,seed: simulate_spike_rates(gain=gain,offset=5.0,
                                                               n_conds=2,n_trials=n_trials,
                                                               seed=seed)

    elif test == 'n_conds':
        values = [2,4,8] if values is None else values
        gen_rates = lambda n_conds,seed: simulate_spike_rates(gain=10.0,offset=5.0,
                                                              n_conds=n_conds,n_trials=1000,
                                                              seed=seed)
        
    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)
        
    method_ = method
            
    # Deal with special-case linear models -- funnel into pev function 
    if method in ['pev','regress','anova1','anova2','anovan']:
        method_ = 'pev'
        # For PEV, additional argument to neural_info() specifying linear model to use
        if method == 'pev': kwargs.update({'model':'anova1'})
        else:               kwargs.update({'model':method})
        
    # For these signed binary methods, reverse default grouping bc in stimulated rates,
    # group1 > group0, but signed info assumes opposite preference
    elif method in ['dprime','d','cohensd', 'auroc','roc','aucroc','auc']:
        if 'groups' not in kwargs: kwargs.update({'groups':[1,0]})
       
    # Expected baseline value for no btwn-condition = 0.5 for AUROC, 0 for other methods
    baseline = 0.5 if method in ['auroc','roc','aucroc','auc'] else 0
                 
    info = np.empty((len(values),n_reps))
        
    for i,value in enumerate(values):        
        for seed in range(n_reps):
            # Generate simulated spike rates with given test value and random seed
            rates,labels = gen_rates(value,seed)

            # For regression model, convert labels list -> design matrix, append intercept term 
            if method == 'regress': labels = dmatrix('1 + C(vbl1,Sum)',{'vbl1':labels})
                            
            info[i,seed] =  neural_info(labels,rates,method=method_,**kwargs)            
            
    # Compute mean and std dev across different reps of simulation            
    sd = info.std(axis=1,ddof=0)
    info = info.mean(axis=1)
    
    if plot:
        plt.figure()
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
        plt.errorbar(values, info, sd, marker='o')
        xlabel = 'n' if test == 'bias' else test
        plt.xlabel(xlabel)
        plt.ylabel("Information (%s)" % method_)
        
    # Determine if test actually produced the expected values
    # 'gain' : Test if information increases monotonically with gain
    if test == 'gain':
        assert (np.sort(info) == info).all(), \
            AssertionError("Information does not increase monotonically with between-condition rate difference")
                                
    # 'n' : Test if information is ~ same for all values of n (unbiased by n)      
    elif test in ['n','n_trials']:
        assert info.ptp() < sd.max(), \
            AssertionError("Information has larger than expected range across n's (likely biased by n)")
        
    # 'bias': Test if information is not > baseline if gain = 0, for varying n
    elif test == 'bias':
        assert ((info - baseline) < sd).all(), \
            AssertionError("Information is above baseline for no rate difference between conditions")
         
    return info, sd