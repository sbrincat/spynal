# -*- coding: utf-8 -*-
"""
info    A module for computing measures of neural information about task/behavioral variables

FUNCTIONS
### High-level information computation wrapper functions ###
neural_info     Wrapper function computes neural information using any given method
neural_info_2groups Wrapper function computes binary neural info with 2 data groups as inputs
neural_info_ngroups Wrapper function computes multi-class neural info with n data groups as inputs

### Multivariate population decoding (classification accuracy) ###
decode          Computes neural info as accuracy of decoding task conds from neural activity

### Shannon mutual information-related functions ###
mutual_info     Computes neural info as Shannon mutual information btwn response and task conds

### Area under receiver operating characteristic curve-related functions ###
auroc           Computes neural info as area under ROC curve

### D-prime/Cohen's d-related functions ###
dprime          Computes neural information as d-prime (Cohen's d)

### Percent explained variance-related functions ###
pev             Computes neural info as percent explained variance (with optional stats)
anova1          Computes PEV and stats using 1-way ANOVA
anova2          Computes PEV and stats using 2-way ANOVA
regress         Computes PEV and stats using 2-way linear regression

### Utility functions ###
patsy_terms_to_columns Returns regress term corresponding to each col in patsy DesignMatrix


DEPENDENCIES
patsy           Python package for describing statistical models


Created on Mon Sep 17 00:05:25 2018

@author: sbrincat
"""
import numpy as np

from scipy.stats import f as Ftest
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from patsy import DesignMatrix

from neural_analysis.utils import unsorted_unique, standardize_array, undo_standardize_array
from neural_analysis.helpers import _has_method


# =============================================================================
# High-level neural information wrapper functions
# =============================================================================
def neural_info(labels, data, axis=0, method='pev', **kwargs):
    """
    Wrapper function to compute mass-univariate neural information about
    some task/behavioral variable(s)

    info = neural_info(labels,data,axis=0,method='pev',**kwargs)

    ARGS
    labels  (n_obs,) | (n_obs,n_terms) array-like. List of labels or design matrix.
            Must be same length as data.shape[0] along dimension <axis>.

    data    (...,n_obs,...) ndarray. Neural data to fit. Axis <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels.

    axis    Int. Axis of data array to perform analysis on, corresponding
            to trials/observations. Default: 0 (first axis)

    method  String. Method to use to compute information. Options:
            'pev' | 'dprime' | 'auroc' | 'mutual_info' | 'decode'
            Default: 'pev'

    **kwargs All other kwargs passed directly to information method function

    RETURNS
    info    (...,n_terms,...). Measure of information in data about labels
            Shape is same as data, with observation axis reduced to length = n_terms.
    """
    method = method.lower()
    info_func = _string_to_info_func(method)
    return info_func(labels,data,axis=axis,**kwargs)


def neural_info_2groups(data1, data2, axis=0, method='pev', **kwargs):
    """
    Wrapper function to compute mass-univariate neural information about
    some binary (ie 2-group) task/behavioral variable, with the inputs being the
    data for each of the two possible conditions: (data1, data2)

    Note: Some methods (dprime, auroc) are faster with this method and it is
    thus preferred, but otherwise the main reason to use it is for convenience
    if your data is already formatted into two groups

    info = neural_info_2groups(data1,data2,axis=0,method='pev',**kwargs)

    ARGS
    data1/2 (...,n_obs1/n_obs2,...) ndarrays of arbitary size except for <axis>.
            Sets of values for each of two distributions to be compared.

    axis    Int. Axis of data array to perform analysis on, corresponding
            to trials/observations. Default: 0 (first axis)

    method  String. Method to use to compute information. Options:
            'pev' | 'dprime' | 'auroc' | 'mutual_info' | 'decode'
            Default: 'pev'

    **kwargs All other kwargs passed directly to information method function

    RETURNS
    info    (...,n_terms,...). Measure of information in data about labels
            Shape is same as data, with observation axis reduced to length = n_terms.
    """
    method = method.lower()

    two_group_methods = ['auroc','roc','aucroc','auc', 'dprime','d','cohensd']

    # Methods that prefer to call 2-group version bc it's faster -- just call 2-group version
    if method in two_group_methods:
        if method in ['dprime','d','cohensd']:                  info_func = _dprime_2groups
        elif method in ['auroc','roc','aucroc','auc']:          info_func = _auroc_2groups
        else:
            raise ValueError("Information method '%s' is not yet supported" % method)

        return info_func(data1, data2, axis=axis, **kwargs)

    # All other methods -- create label list, concatenate data and call (label,data) version
    else:
        info_func = _string_to_info_func(method)

        n1 = data1.shape[axis]
        n2 = data2.shape[axis]
        assert (n1 != 0) and (n2 != 0), \
            "Data contains no observations (trials) for one or more groups"

        labels = np.hstack((np.zeros((n1,),dtype='uint8'), np.ones((n2,),dtype='uint8')))

        return info_func(labels, np.concatenate((data1,data2), axis=axis), axis=axis, **kwargs)


def neural_info_ngroups(*args, axis=0, method='pev', **kwargs):
    """
    Wrapper function to compute mass-univariate neural information about
    some multi-class (ie n group) variable, with the inputs being the
    data for each of the possible conditions: (data1, data2, ..., data_k)

    Note: Some methods (dprime, auroc) are designed for only binary comparisons
    and will raise an error if you try to call them with this function

    info = neural_info_ngroups(data1,data2,...,data_k,axis=0,method='pev',**kwargs)

    ARGS
    data1...k (...,n_obs1/k,...) ndarrays of arbitary size except for <axis>.
            Sets of values for each data distribution to be compared.

    axis    Int. Axis of data array to perform analysis on, corresponding
            to trials/observations. Default: 0 (first axis)

    method  String. Method to use to compute information. Options:
            'pev' | 'dprime' | 'auroc' | 'mutual_info' | 'decode'
            Default: 'pev'

    **kwargs All other kwargs passed directly to information method function

    RETURNS
    info    (...,n_terms,...). Measure of information in data about difference between
            data1 vs data2 vs ... data_k
            Shape is same as data, with observation axis reduced to length = n_terms.
    """
    # TODO Add 'mutual_information','mutual_info' when mutual_info is updated for multi-class
    n_group_methods = ['decode','decoder','decoding', 'pev']

    n_groups = len(args)

    assert (n_groups == 2) or (method in n_group_methods), \
        "Must specify information 'method' that works on multi-class problems ('decode'|'pev')"

    info_func = _string_to_info_func(method)

    # Find number of observations (trials) in each group's data
    n = [args[j].shape[axis] for j in range(n_groups)]
    # Create list of integer labels, with n[group] values=group in each
    # ie [0,0,0,...,1,1,1,...,k,k,k....], where k = n_groups
    labels = np.hstack([j*np.ones((n[j],),dtype='uint8') for j in range(n_groups)])

    return info_func(labels, np.concatenate(args, axis=axis), axis=axis, **kwargs)


def _string_to_info_func(method):
    """ Converts string specifier to function for computing neural information """
    if method == 'pev':                                     return pev
    elif method in ['dprime','d','cohensd']:                return dprime
    elif method in ['auroc','roc','aucroc','auc']:          return auroc
    elif method in ['mutual_information','mutual_info']:    return mutual_info
    elif method in ['decode','decoder','decoding']:         return decode
    else:
        raise ValueError("Information method '%s' is not yet supported" % method)


# =============================================================================
# Population decoding (classification) analysis
# =============================================================================
def decode(labels, data, axis=0, feature_axis=1, decoder='LDA', cv='auto', seed=None,
           groups=None, as_pct=False, return_stats=False, stats=None, **kwargs):
    """
    Mass-multivariate population decoding analysis using given classifier method

    accuracy = decode(labels,data,axis=0,feature_axis=1,decoder='LDA',cv='auto',seed=None,
                      groups=None,as_pct=False,return_stats=False,stats=None,**kwargs)

    accuracy,stats = decode(labels,data,axis=0,feature_axis=1,decoder='LDA',cv='auto',seed=None,
                            groups=None,as_pct=False,return_stats=True,stats=None,**kwargs)

    INPUTS
    labels  (n_obs,) ndarray. Labels/target values for each trial to predict

    data    (...,n_obs,...,n_features,...) ndarray. Neural data to decode from.
            Arbitrary shape, but <axis> should correspond to observations (trials) and
            <feature_axis> should correspond to decoder features (eg neural channels),
            while rest of axis(s) can be any independent data series (time points,
            frequencies, etc.) that are analyzed separately (ie separate decoder fit
            and evaluated at each time point, frequency, etc.).

    axis    Int. Axis of data corresponding to distinct observations/trials. Default: 0 (1st axis)

    feature_axis Int. Axis of data corresponding to decoder features (usually distinct neural
            channels/electrodes/units). Default: 1 (2nd axis)

    decoder String | sklearn classifier object. Decoding classifier method to use.
            Can input either as a string specifier or as a scikit-learn classifier object instance.
            'LDA' :     Linear discriminant analysis using
                        sklearn.discriminant_analysis.LinearDiscriminantAnalysis
                        Unlike scikit's empirical prior, sets a uniform prior unless specified
                        otherwise (bc any imbalance in emprical probabilities of task conditions is
                        usually happenstance, not predictive and not something we want to use)
            'logistic': Logistic regression using sklearn.linear_model.LogisticRegression
                        Sets penalty='none' unless specified otherwise in kwargs, unlike scikit's
                        default 'l2' (L2 regularization) bc seems safer to make users opt-in.
            'SVM' :     Support vector machine classification using sklearn.svm.SVC
                        Sets kernel='linear' unless specified otherwise in kwargs, unlike scikit's
                        default 'rbf' (nonlinear radial basis functions) bc linear classifiers are
                        much more common in the field (and safer to make users opt-in to nonlinear)

            Can use custom objects, but must follow sklearn API (ie has 'fit' and 'score' methods).

    cv      String | sklearn.model_selection "Splitter" object. Determines how cross
            validation is done. Set = 'auto' for default. Set = None for no cross-validation.
            Can use custom objects, but must follow "Splitter' object API (ie has 'split' method).
            Default: StratifiedKFold(n_splits=5,shuffle=True)

    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    groups  Array-like. Which group labels from <labels> to use. Useful for computing information
            on subset of groups/classes in labels.
            Default: unique(labels) (all distinct values in <labels>)

    as_pct  Bool. Set=True to return decoding accuracy as a percent (range ~ 0-100).
            Set=False [default] to return accuracy as a proportion (range ~ 0-1)

    return_stats Bool. Set=True to return additional classifier stats. Default: False

    stats   String | List of strings. List of additional classifier stats to return:
            'predict' : Predicted class for each trial/observation
            'prob' :    Posterior probability for each class, for each trial/observation
            Default: If return_stats=True, stats defaults to ['predict','prob']

    **kwargs All other kwargs passed directly to decoding object constructor

    OUTPUTS
    accuracy ndarray. Decoding accuracy. Shape is same as input data, but with <axis> and
            <feature_axis> reduced to length 1. Accuracy given as proportion or percent
            correct, depending on value of <as_pct>. Chance = [100 *] 1/n_classes.

    stats   Dict. Optional output. Additional per-trial decoding-related stats, as requested in
            input argument <stats>. May include the following:
            'predict' : (...,_n_obs,...,1,...) ndarray. Predicted class for each observation/trial.
                        Same shape as accuracy, but <axis> has length n_obs.
            'prob' :    (...,_n_obs,...,n_classes,...) ndarray. Posterior probabilty for each class
                        and each observation (trial). Same shape as accuracy, but <axis> has length
                        n_obs and <feature_axis> has length n_classes.

    REFERENCE
    https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    labels = np.asarray(labels)
    data = np.asarray(data)

    if axis < 0:            axis = data.ndim + axis
    if feature_axis < 0:    feature_axis = data.ndim + feature_axis

    # Swap array axes so trials/observations axis is 1st and features 2nd (n_obs,n_features,...)
    if (axis == 1) and (feature_axis == 0):
        data = np.swapaxes(data,feature_axis,axis)
    else:
        if axis != 0:           data = np.moveaxis(data,axis,0)
        if feature_axis != 1:   data = np.moveaxis(data,feature_axis,1)
    data_ndim = data.ndim
    data_shape = data.shape

    # Standardize data array to shape (n_obs,n_features,n_data_series)
    if data_ndim > 3:       data = data.reshape((data.shape[0],data.shape[1],-1))
    elif data_ndim == 2:    data = data[:,:,np.newaxis]
    elif data_ndim == 1:    data = data[:,np.newaxis,np.newaxis] # Weird usage, but ok

    # Find set of unique group/class labels in <labels> if not explicitly set
    if groups is None:
        groups = np.unique(labels)

    # If groups set in args, remove any observations not represented in <groups>
    else:
        idxs = np.in1d(labels,groups)
        if idxs.sum() != data.shape[0]:
            labels  = labels[idxs]
            data    = data[idxs,...]

    n_obs,n_features,n_series = data.shape
    n_classes = len(groups)

    if stats is not None: return_stats = True   # If requested stats list input, return stats
    if return_stats:
        if stats is None:  stats = ['predict','prob']
        elif isinstance(stats,str): stats = [stats]
    else:
        stats = range(0)    # Kludge -- set this to shunt off and skip "for stats" loops below

    # Convert string specifier to sklearn classifier object
    if isinstance(decoder,str):
        decoder = decoder.lower()

        if decoder in ['lda','lineardiscriminant']:
            # If not specified otherwise, set uniform prior
            if 'priors' not in kwargs: kwargs['priors'] = (1/n_classes)*np.ones((n_classes,))
            decoder = LinearDiscriminantAnalysis(**kwargs)

        elif decoder in ['logistic','logisticregression']:
            # If not specified otherwise, set uniform prior
            if 'penalty' not in kwargs: kwargs['penalty'] = 'none'
            decoder = LogisticRegression(**kwargs)

        elif decoder in ['svm','svc']:
            # If not specified otherwise, set linear kernel
            if 'kernel' not in kwargs: kwargs['kernel'] = 'linear'
            decoder = SVC(**kwargs)

        else:
            raise ValueError("Unsupported option '%s' given for decoding method" % decoder)

    else:
        # Ensure we actually have an sklearn(-like) classifier object
        assert _has_method(decoder,'fit') and _has_method(decoder,'score'), \
            TypeError("Unsupported type (%s) for decoder. Use string | scikit classifier object"
                      % type(decoder))
        # Ensure no additional parameters were passed that would be ignored
        assert len(kwargs) == 0, \
            TypeError("No additional arguments allowed here if using custom decoder function")

    # Convert string specifier to scikit cross-validation object, including default
    if isinstance(cv,str):
        cv = cv.lower()
        # Default <cv>: shuffled 5-fold StratifiedKFold
        if cv in ['default','auto','stratifiedkfold']:
            cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        elif cv == 'none':
            cv = None
        else:
            raise ValueError("Unsupported value '%s' given for cv" % cv)

    else:
        # Ensure we actually have an sklearn(-like) cross-validator object
        assert _has_method(cv,'split') or (cv is None), \
            TypeError("Unsupported type (%s) for cv. Use string | scikit model_selection object"
                        % type(cv))

    if return_stats:
        stats = {stat:None for stat in stats}
        for stat in stats:
            if stat == 'predict':
                stats[stat] = np.empty((n_obs,1,n_series),dtype=labels.dtype)
                method = 'predict'
            elif stat == 'prob':
                stats[stat] = np.empty((n_obs,n_classes,n_series))
                method = 'predict_proba'
            else:
                raise ValueError("Unsupported stat '%s' requested" % stat)

            assert _has_method(decoder,method), \
                TypeError("Decoder object does not contain '%s' method for requested '%s' stat"
                          % (method,stat))

    # Iterate analysis over all cross-validation train/test data splits
    if cv is not None:
        acc_folds = np.empty((n_series,cv.n_splits))

        # todo  More sklearn way of doing this?
        for i_fold,(train_idxs,test_idxs) in enumerate(cv.split(data,labels)):
            for i_series in range(n_series):
                # Fit model to training split
                decoder = decoder.fit(data[train_idxs,:,i_series], labels[train_idxs])
                # Evaluate model on testing split
                acc_folds[i_series,i_fold] = \
                    decoder.score(data[test_idxs,:,i_series], labels[test_idxs])

                for stat in stats:
                    if stat == 'predict':
                        stats[stat][test_idxs,0,i_series] = \
                            decoder.predict(data[test_idxs,:,i_series])
                    elif stat == 'prob':
                        tmp = decoder.predict_proba(data[test_idxs,:,i_series])
                        stats[stat][test_idxs,:,i_series] = tmp

        # Take the mean across all cross-validation folds
        accuracy = np.mean(acc_folds, axis=1)

    # Run analysis without any cross-validation
    else:
        accuracy = np.empty((n_series,))

        for i_series in range(n_series):
            # Train/test on full data
            decoder = decoder.fit(data[:,:,i_series], labels)
            accuracy[i_series] = decoder.score(data[:,:,i_series], labels)

    if as_pct: accuracy = 100.0*accuracy

    # Insert 2 singleton axis as 1st axes of array to replace trial,channel axes
    accuracy = accuracy[np.newaxis,np.newaxis,...]

    # Reshape "data series" axis back to original dimensionality
    if data_ndim > 3:
        accuracy = accuracy.reshape((1,1,*data_shape[2:]))
        for stat in stats:
            if stat == 'predict':
                stats[stat] = stats[stat].reshape((n_obs,*data_shape[2:]))
            elif stat == 'prob':
                stats[stat] = stats[stat].reshape((n_obs,n_classes,*data_shape[2:]))

    # Move/swap array axes to original locations
    if (axis == 1) and (feature_axis == 0):
        data = np.swapaxes(data,axis,feature_axis)
        for stat in stats:
            if stat == 'predict':   stats[stat] = np.swapaxes(stats[stat],axis,feature_axis)
            elif stat == 'prob':    stats[stat] = np.swapaxes(stats[stat],axis,feature_axis)
    else:
        if axis != 0:
            data = np.moveaxis(data,0,axis)
            for stat in stats:
                if stat == 'predict':   stats[stat] = np.moveaxis(stats[stat],0,axis)
                elif stat == 'prob':    stats[stat] = np.moveaxis(stats[stat],0,axis)
        if feature_axis != 1:
            data = np.moveaxis(data,1,feature_axis)
            for stat in stats:
                if stat == 'predict':   stats[stat] = np.moveaxis(stats[stat],1,feature_axis)
                elif stat == 'prob':    stats[stat] = np.moveaxis(stats[stat],1,feature_axis)

    if data_ndim <= 2:
        accuracy = accuracy.item()
        for stat in stats:  stats[stat] = stats[stat].squeeze()

    if return_stats:    return accuracy, stats
    else:               return accuracy


# =============================================================================
# Mutual information analysis
# =============================================================================
def mutual_info(labels, data, axis=0, bins=None, resp_entropy=None, groups=None):
    """
    Mass-univariate mutual information between set of discrete-valued (or discretized)
    neural responses and categorical experimental conditions (often referred to as "stimuli"
    in theoretical treatments).

    NOTE: Currently only 2-class (binary) conditions are supported

    info = mutual_info(labels,data,axis=0,bins='fd',resp_entropy=None,groups=None)

    Computes Shannon mutual information using standard equation (cf. Dayan & Abbott, eqn. 4.7):
    I = H - Hnoise = -Sum(p(r)*log(p(r)) + Sum(p(cat)*p(r|s)*log(p(r|s)))

    where H = total response entropy, Hnoise = noise entropy, p(r) is response probability
    distribution, and p(r|s) is conditional probability of response given experimental condition

    info = 0 indicates no mutual information between responses and experimental conditions
    info = 1 indicates maximum possible information btwn responses and conditions

    Computation is performed over trial/observation <axis>, in mass-univariate
    fashion across data series in all other data dimensions (channels, time points,
    frequencies, etc.).

    ARGS
    labels      (n_obs,) array-like. List of categorical group labels labelling observations
                from each group. NOTE: Should be only two groups represented, unless sub-selecting
                two groups using <groups> argument.

    data        (...,n_obs,...) ndarray. Data values to compute mutual information with labels.
                If it is not discrete-valued (eg spike counts), it will be discretized using 'bins'

    axis        Scalar. Axis of data array to perform analysis on, corresponding
                to trials/observations. Default: 0 (first axis)

    bins        (n_bins,2) array-like | string. Non-integer data must be binned for
                mutual information computation. Bins can be given either explicitly, as an
                array of bin [left,right] edges, or as a string indicating the type of
                binning rule to use in np.histogram_bin_edges() (see there for details).
                Default: 'fd' (Freedman–Diaconis rule: bin width = 2*IQR(data)/cuberoot(n))
                Data is binned only if it is non-integer-valued or if a value is input for bins.

    resp_entropy   (...,1,...). Total response entropy. Can optionally compute and input
                this to save repeated calculations (eg for distinct contrasts on same data)

    groups      (2,) array-like. Which group labels from <labels> to use.
                Useful for computing info btwn pairs of groups in data with > 2 groups,
                Default: unique(labels) (all distinct values in <labels>)

    RETURNS
    info        (...,1,...) ndarray. Mutual information between responses
                and experimental conditions (in bits).

    REFERENCE   Dayan & Abbott, _Theoretical neuroscience_ ch. 4.1
    (binning)   https://stats.stackexchange.com/questions/179674/number-of-bins-when-computing-mutual-information/181195
    (binning)   https://en.wikipedia.org/wiki/Freedman-Diaconis_rule
    """
    # TODO  Recode for > 2 groups
    labels = np.asarray(labels)

    # Reshape data array -> (n_observations,n_data_series) matrix
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    if resp_entropy is not None:
        resp_entropy,_ = standardize_array(resp_entropy, axis=axis, target_axis=0)

    n_series = data.shape[1] if data.ndim > 1 else 1

    # Find set of unique group labels in <labels> if not explicitly set
    if groups is None:
        groups = np.unique(labels)

    # If groups set in args, remove any observations not represented in <groups>
    else:
        idxs = np.in1d(labels,groups)
        if idxs.sum() != data.shape[0]:
            labels  = labels[idxs]
            data    = data[idxs,...]

    assert len(groups) == 2, \
        "mutual_info: computation only supported for 2 groups (%d given)" % len(groups)

    # Bin-discretize data if it is not already integer-valued OR if value is input for bins
    do_bins = (bins is not None) or \
              (np.issubdtype(data.dtype,float) and not np.allclose(np.round(data), data))

    if do_bins:
        # If bins are given explicitly, reshape them as needed
        if isinstance(bins, (list, tuple, np.ndarray)):
            bins = np.asarray(bins)
            assert (bins.ndim == 2) and (bins.shape[1] == 2), \
                ValueError("bins must be given as (n_bins,2) array of [left,right] edges")

            # Convert bins to format expected by np.digitize = edges of all bins in 1 series
            bins_ = np.hstack((bins[:,0],bins[-1,-1]))

        # Otherwise, use some heuristic rule to set bins based on data
        else:
            if bins is None: bins = 'fd'
            bins_ = np.histogram_bin_edges(data.flatten(), bins=bins)

        # Bin the data. Note: this actually returns bin numbers for each datapoint,
        # but MI treats values categorically, so doesn't matter
        data = np.digitize(data, bins_)

    idxs1 = labels == groups[0]
    idxs2 = labels == groups[1]
    n1 = idxs1.sum()
    n2 = idxs2.sum()
    N  = n1 + n2

    assert (n1 != 0) and (n2 != 0), \
        "mutual_info: Data contains no observations (trials) for one or more groups"

    # Stimulus probabilities
    Ps1 = n1/N                  # P(stimulus==s1)
    Ps2 = n2/N                  # P(stimulus==s2)

    info = np.empty((n_series,))

    for i_series in range(n_series):
        data_series = data[:,i_series] if n_series > 1 else data.squeeze()

        # Compute total response entropy H (if not input)
        if resp_entropy is None:
            # Response probability distributions
            _,counts    = np.unique(data_series, return_counts=True)
            Pr          = counts / N            # P(response==r)

            # Response entropy (Dayan & Abbott, eqn. 4.3)
            H   = entropy(Pr)

        else:
            H   = resp_entropy[i_series]

        # Conditional response|stimulus probability distributions (eg, Prs1 is p(rateR|stim1))
        # Note: conditional probs of 0 are automatically eliminated, since by convention 0*log0=0
        _,counts    = np.unique(data_series[idxs1], return_counts=True)
        Prs1        = counts / counts.sum()                 # P(response==r | stimulus1)
        _,counts    = np.unique(data_series[idxs2], return_counts=True)
        Prs2        = counts / counts.sum()                 # P(response==r | stimulus2)

        # Conditional entropies for stimuli 1 & 2 (Dayan & Abbott, eqn. 4.5)
        Hs1     = entropy(Prs1)
        Hs2     = entropy(Prs2)

        # Noise entropy (Dayan & Abbott, eqn. 4.6)
        Hnoise  = Ps1*Hs1 + Ps2*Hs2

        # Mutual information is total response entropy - noise entropy (Dayan & Abbott, eqn. 4.7)
        info[i_series] = H - Hnoise

    # Pre-pend singleton axis to make info expected (1,n_series) shape
    info = undo_standardize_array(info[np.newaxis,:], data_shape, axis=axis, target_axis=0)

    return info


mutual_information = mutual_info
""" Aliases function pev as percent_explained_variance """

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

    Note: This is actually a wrapper around _auroc_2groups(), which accepts
    two data distributions as arguments, and is faster.

    ARGS
    labels      (n_obs,) array-like. List of group labels labelling observations from
                each group. Should be only two groups represented, unless sub-selecting
                two groups using <groups> argument.

    data        (...,n_obs,...) ndarray. Data values for both distributions to be compared.

    axis        Scalar. Axis of data array to perform analysis on, corresponding
                to trials/observations. Default: 0 (first axis)

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

    # Split data into 2 groups and use _auroc_2groups() to actually do computation
    return _auroc_2groups(data.compress(labels == groups[0], axis=axis),
                          data.compress(labels == groups[1], axis=axis),
                          axis=axis, signed=signed)


def _auroc_2groups(data1, data2, axis=0, signed=True):
    """
    Version of auroc() that accepts two data distributions
    auroc() actually calls this for its computation, bc it's faster
    """
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]

    assert (n1 != 0) and (n2 != 0), \
        "auroc: Data contains no observations (trials) for one or more groups"

    # Reshape data arrays -> (n_observations,n_data_series) matrix
    data1, data1_shape = standardize_array(data1, axis=axis, target_axis=0)
    data2, data2_shape = standardize_array(data2, axis=axis, target_axis=0)

    assert data1.ndim == data2.ndim, "auroc: data1,2 must have same dimensionality"
    if data1.ndim > 1:
        assert (*data1_shape[:axis],*data1_shape[axis+1:]) == \
               (*data2_shape[:axis],*data2_shape[axis+1:]), \
            "auroc: data1,2 must have same number of data series (timepts,freqs,channels,etc.)"

    n1 = data1.shape[0]
    n2 = data2.shape[0]

    n_series = data1.shape[1] if data1.ndim > 1 else 1

    roc_area = np.empty((n_series,))

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

    # Pre-pend singleton axis to make info expected (1,n_series) shape
    roc_area = undo_standardize_array(roc_area[np.newaxis,:], data1_shape, axis=axis, target_axis=0)

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

    Note: This is actually a wrapper around _dprime_2groups(), which accepts
    two data distributions as arguments, and is faster.

    ARGS
    labels      (n_obs,) array-like. List of group labels labelling observations from
                each group. Should be only two groups represented, unless sub-selecting
                two groups using <groups> argument.

    data        (...,n_obs,...) ndarray. Data values for both distributions to be compared.

    axis        Scalar. Axis of data array to perform analysis on, corresponding
                to trials/observations. Default: 0 (first axis)

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

    # Split data into 2 groups and use _dprime_2groups() to actually do computation
    return _dprime_2groups(data.compress(labels == groups[0], axis=axis),
                           data.compress(labels == groups[1], axis=axis),
                           axis=axis, signed=signed)


def _dprime_2groups(data1, data2, axis=0, signed=True):
    """
    Version of dprime() that accepts two data distributions
    dprime() actually calls this for its computation, bc it's faster
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

    # For scalar info (vector data), extract value from scalar array -> float for output
    if d.size == 1: d = d.item()

    return d


# =============================================================================
# Percent explained variance (PEV) analysis
# =============================================================================
def pev(labels, data, axis=0, model=None, omega=True, as_pct=True, return_stats=False, **kwargs):
    """
    Mass-univariate percent explained variance (PEV) analysis.

    Computes the percentage (or proportion) of variance explained in data by
    predictors in design matrix/list of labels, using one of a few types of linear models.

    exp_var = pev(labels,data,axis=0,model=None,omega=True,as_pct=True,
                  return_stats=False,**kwargs)

    exp_var,stats = pev(labels,data,axis=0,model=None,omega=True,as_pct=True,
                        return_stats=False,**kwargs)

    ARGS
    labels  (n_obs,n_terms) array-like | patsy DesignMatrix object. Design matrix
            (group labels for ANOVA models, or regressors for regression model) for
            each observation (trial). labels.shape[0] must be same length as observation
            <axis> of data.

    data    (...,n_obs,...) ndarray. Data to fit with linear model. Axis <axis> should
            correspond to observations (trials), while any other axes can be any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels <labels>.

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    model   String. Type of linear model to fit, in order to compute PEV.
            'anova1'    : 1-way ANOVA model (labels is (n_obs,) vector)
            'anova2'    : 2-way ANOVA model (labels must be a (n_obs,2) array)
            'anovan'    : n-way ANOVA model (labels must be a (n_obs,n_terms) array)
            'regress'   : linear regression model (labels is (n_obs,nModelParams) array)
            Default: we attempt to infer from <labels>. Safest to set explicitly.

    omega   Bool. If True, uses bias-corrected omega-squared formula for PEV,
            otherwise uses eta-squared/R-squared formula, which is positively biased.
            Default: True

    as_pct  Bool. Set=True [default] to return PEV as a percent (range ~ 0-100).
            Otherwise PEV returned as a proportion (range ~ 0-1)

    return_stats Bool. Set=True to return several stats on model fit (eg F-stat,p)
            in addition to PEV. Otherwise, just returns PEV. Default: False

    **kwargs Passed directly to model function. See those for details.

    RETURNS
    exp_var (...,n_terms,...). Percent (or proportion) of variance in data explained by labels
            Shape is same as data, with observation axis reduced to length = n_terms.

    stats   Dict. If <return_stats> set, statistics on each fit also returned.
            See model function for specific stats returned by each.
    """
    # TODO Add anovan model
    if not isinstance(labels,DesignMatrix): labels = np.asarray(labels)

    # Attempt to infer proper linear model based on labels
    if model is None:
        # If labels is vector-valued, assume 1-way ANOVA model
        if (labels.ndim == 1) or (labels.shape[1] == 1):    model = 'anova1'
        # If labels has constant/intercept term (column of all 1's), assume regression model
        elif (labels == 1).all(axis=0).any():               model = 'regress'
        # If labels has > 3 columns, assume n-way ANOVA
        # TODO ADD: elif labels.shape[1] > 3:                   model = 'anovan'
        # Otherwise, could be ANOVA2, ANOVAn, regress ... dangerous to assume
        else:
            raise ValueError("Could not determine appropriate linear model.\n" \
                             "Please set explicitly using <model> argument.")

        print("Assuming '%s' linear model based on given <labels> labels/design matrix" % model)

    model = model.lower()

    # Compute PEV based on 1-way ANOVA model
    if model == 'anova1':
        return anova1(labels,data,axis=axis,omega=omega,as_pct=as_pct,
                      return_stats=return_stats,**kwargs)
    # Compute PEV based on 2-way ANOVA model
    elif model == 'anova2':
        return anova2(labels,data,axis=axis,omega=omega,as_pct=as_pct,
                      return_stats=return_stats,**kwargs)
    # Compute PEV based on 2-way ANOVA model
    elif model == 'regress':
        return regress(labels,data,axis=axis,omega=omega,as_pct=as_pct,
                       return_stats=return_stats,**kwargs)
    else:
        raise ValueError("'%s' model is not supported for computing PEV" % model)

percent_explained_variance = pev
""" Aliases function pev as percent_explained_variance """


def anova1(labels, data, axis=0, omega=True, groups=None, gm_method='mean_of_obs',
           as_pct=True, return_stats=False):
    """
    Mass-univariate 1-way ANOVA analyses of one or more data vector(s)
    on single list of group labels

    exp_var = anova1(labels,data,axis=0,omega=True,groups=None,gm_method='mean_of_obs',
                     as_pct=True,return_stats=False)

    exp_var,stats = anova1(labels,data,axis=0,omega=True,groups=None,gm_method='mean_of_obs',
                           as_pct=True,return_stats=False)

    ARGS
    labels  (n_obs,) array-like. Group labels for each observation (trial),
            identifying which group/factor level each observation belongs to.
            Number of rows(labels.shape[0] = n_obs) be same length as data.shape[axis].

    data    (...,n_obs,...) ndarray. Data to fit with ANOVA model. <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels labels.

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    omega   Bool. Determines formula for calculating PEV.  Default: True
            True  : Bias-corrected omega-squared PEV formula [default]
            False : Standard eta-squared formula. Positively biased for small N.

    groups  (n_groups,) array-like. Which group labels from <labels> to use.
            Useful for enforcing a given order to groups (reflected in mu's),
            or for computing PEV btwn subsets of groups in data.
            Default: unique(labels) (all distinct values in <labels>)

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
    exp_var (..,1,...). Percent (or proportion) of variance in data explained by labels.
            Shape is same as data, with observation axis reduced to length 1.

    stats   Dict. If <return_stats> set, statistics on each fit also returned:
        p   (...,1,...). F-test p values for each datapoint. Same shape as exp_var.
        F   (...,1,...). F-statistic for each datapoint. Same shape as exp_var.
        mu  (...,n_groups,...). Group mean for each group/level
        n   (...,n_groups,). Number of observations (trials) in each group/level
    """
    if not isinstance(labels,DesignMatrix): labels = np.asarray(labels)

    assert (labels.ndim == 1) or (labels.shape[1] == 1), \
            "labels should have only a single column for anova1 model (it has %d)" \
            % labels.shape[1]

    # Reshape data array data -> (n_observations,n_data_series) matrix
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)

    # Find set of unique group labels in list of labels
    if groups is None:
        groups = np.unique(labels)

    # Remove any observations not represented in <groups>
    else:
        idxs = np.in1d(labels,groups)
        if idxs.sum() != data.shape[0]:
            labels  = labels[idxs]
            data    = data[idxs,:]

    n_groups = len(groups)
    n_obs,n_series = data.shape

    assert labels.shape[0] == n_obs, \
            "labels and data array should have same number of rows (%d != %d)" \
            % (labels.shape[0], n_obs)

    # Compute mean for each group (and for each data series)
    n = np.empty((n_groups,))
    mu = np.empty((n_groups,n_series))

    for i_group,group in enumerate(groups):
        group_idxs   = labels == group
        n[i_group]   = group_idxs.sum()   # Number of observations for given group
        mu[i_group,:]= data[group_idxs,:].mean(axis=0)  # Group mean for given group

    # Compute grand mean across all observations (for each data series)
    if gm_method == 'mean_of_obs':        grand_mean = data.mean(axis=0)
    elif gm_method == 'mean_of_means':    grand_mean = mu.mean(axis=0)

    # Total Sums of Squares
    SS_total = ((data - grand_mean)**2).sum(axis=0)

    # Groups Sums of Squares
    SS_groups = np.zeros((1,n_series))
    for i_group in range(n_groups):
        # Group Sum of Squares for given group
        SS_groups += n[i_group]*(mu[i_group,:] - grand_mean)**2

    # Compute degrees of freedom and Mean Squares needed for PEV and/or F-stat formulas
    if omega or return_stats:
        df_groups= n_groups - 1         # Groups degrees of freedom
        df_error = n_obs-1 - df_groups  # Error degrees of freedom
        MS_error = (SS_total - SS_groups) / df_error   # Error mean square

    # PEV,F strictly undefined when no data variance (div by 0).
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

    exp_var = undo_standardize_array(exp_var, data_shape, axis=axis, target_axis=0)

    if not return_stats:
        return exp_var

    # Calculate F-statistic and perform F-test to determine p value for all data points
    else:
        MS_groups= SS_groups / df_groups    # Groups mean square
        F       = MS_groups / MS_error      # F statistic
        F[:,undefined] = 0                  # Set F = 0 for data w/ data variance = 0
        p       = Ftest.sf(F,df_groups,df_error) # p value for given F stat

        F   = undo_standardize_array(F, data_shape, axis=axis, target_axis=0)
        p   = undo_standardize_array(p, data_shape, axis=axis, target_axis=0)
        mu  = undo_standardize_array(mu, data_shape, axis=axis, target_axis=0)

        stats   = {'p':p, 'F':F, 'mu':mu, 'n':n}
        return exp_var, stats


def anova2(labels, data, axis=0, interact=None, omega=True, partial=False, total=False,
           gm_method='mean_of_obs', as_pct=True, return_stats=False):
    """
    Mass-univariate 2-way ANOVA analyses of one or more data vector(s)
    on single set of group labels

    exp_var = anova2(labels,data,axis=0,interact=None,omega=True,partial=False,total=False,
                     gm_method='mean_of_obs',as_pct=True,return_stats=False)

    exp_var,stats = anova2(labels,data,axis=0,interact=None,omega=True,partial=False,total=False,
                           gm_method='mean_of_obs',as_pct=True,return_stats=False)

    ARGS
    labels  (n_obs,n_terms=2|3) array-like. Group labels for each observation (trial),
            identifying which group/factor level each observation belongs to.
            Can either set interaction term labels for column 3, or set
            interact==True and we will auto-generate interaction term
            Number of rows(labels.shape[0] = n_obs) be same length as data.shape[axis].

    data    (...,n_obs,...) ndarray. Data to fit with ANOVA model. <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels.

    axis    Int. Data axis corresponding to distinct observations/trials. Default: 0

    interact Bool. Determines whether an interaction term is included in model.
            If set, but but no 3rd entry is given in labels, we auto-generate an interaction
            term based on all unique combinations of levels in labels[:,0] & labels[:,1].
            Default: true iff labels has 3rd column/entry (for interaction)

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
    exp_var (n_terms,...). Percent (or proportion) of variance in data explained by labels.
            Shape is same as data, with observation axis reduced to length = n_terms.

    stats   Dict. If <return_stats> set, statistics on each fit also returned:
        p   (...,n_terms,...). F-test p values for each datapoint. Same shape as exp_var.
        F   (...,n_terms,...). F-statistic for each datapoint. Same shape as exp_var.
        mu  [n_terms] list of (...,n_groups,...). Group mean for each group (level),
            in a separate list element for each model term (b/c n_groups not same).
        n   [n_terms] list of (...,n_groups,...). Number of observations (trials)
            in each group/level, in a separate list element for each model term.

    REFERENCE   Zar _Biostatistical Analysis_ 4th ed.
    """
    # TODO Add <groups> arg with list of group labels to use?
    if not isinstance(labels,DesignMatrix): labels = np.asarray(labels)

    # Reshape data array data -> (n_observation,n_data_series) matrix
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    n_obs,n_series= data.shape

    # If interaction term is given in labels, its an interaction model
    if interact is None:  interact = labels.shape[1] == 3
    n_terms      = 2 + interact

    assert labels.shape[0] == n_obs, \
            "labels and data array should have same number of rows (%d != %d)" \
            % (labels.shape[0], n_obs)
    assert labels.shape[1] in [2,3], \
            "labels should have 2 or 3 columns for anova2 model (it has %d)" \
            % labels.shape[1]

    # If interaction term is requested, but not provided, create one here
    if interact and labels.shape[1] == 2:
        labels = np.concatenate((labels, np.zeros((n_obs,1),dtype=labels.dtype)), axis=1)
        # All unique combinations of factor 1 & 2
        group_pairs = unsorted_unique(labels[:,0:2], axis=0)
        for i_pair,pair in enumerate(group_pairs):
            # Find and label all observations (trials) with given pair of groups/levels
            idxs = np.all(labels[:,0:2] == pair, axis=1)
            labels[idxs,2] = i_pair

    groups = [np.unique(labels[:,i_term]) for i_term in range(n_terms)]
    n_groups = np.asarray([len(term_groups) for term_groups in groups])

    # Compute means for each group (factor level) in dataset
    n  = []
    mu = []
    for i_term in range(n_terms):
        n.append(np.zeros((n_groups[i_term],)))
        mu.append(np.zeros((n_groups[i_term],n_series)))

        for i_group,group in enumerate(groups[i_term]):
            group_idxs = labels[:,i_term] == group           # Find all obs for current group
            n[i_term][i_group]    = np.sum(group_idxs)  # n for given group
            # Group means for given group
            mu[i_term][i_group,:] = np.mean(data[group_idxs,:], axis=0)

    # Compute grand mean across all observations (for each data series)
    if gm_method == 'mean_of_obs':
        grand_mean = np.mean(data, axis=0)         # Grand mean of all data points
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

    SS_total = np.sum((data - grand_mean)**2, axis=0)  # Total Sum of Squares
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

    # PEV,F strictly undefined when no data variance (div by 0).
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

    exp_var = undo_standardize_array(exp_var, data_shape, axis=axis, target_axis=0)

    # Append summed PEV across all model terms to end of term axis
    if total:   exp_var = np.concatenate((exp_var, np.sum(exp_var,axis=0,keepdims=True)), axis=0)

    if not return_stats:
        return exp_var

    # Calculate F-statistic and perform F-test to determine p value for all data points
    else:
        MS_groups= SS_groups/df_groups  # Groups Mean Squares for each term
        F       = MS_groups/MS_error    # F statistics for each term
        F[:,undefined] = 0              # Set F = 0 for data w/ data variance = 0
        p       = Ftest.sf(F,df_groups,df_error)    # p value for given F stat

        F   = undo_standardize_array(F, data_shape, axis=axis, target_axis=0)
        p   = undo_standardize_array(p, data_shape, axis=axis, target_axis=0)
        mu  = [undo_standardize_array(mu[term], data_shape, axis=axis, target_axis=0)
               for term in range(n_terms)]

        stats   = {'p':p, 'F':F, 'mu':mu, 'n':n}
        return exp_var, stats


def regress(labels, data, axis=0, col_terms=None, omega=True, constant=True,
            partial=False, total=False, as_pct=True, return_stats=False):
    """
    Mass-univariate ordinary least squares regression analyses of one or more
    data vector(s) on single design matrix

    exp_var = regress(labels,data,axis=0,col_terms=None,omega=True,constant=True,
                      partial=False,total=False,as_pct=True,return_stats=False)

    exp_var,stats = regress(labels,data,axis=0,col_terms=None,omega=True,constant=True,
                            partial=False,total=False,as_pct=True,return_stats=False)

    ARGS
    labels  (n_obs,n_params) array-like | patsy DesignMatrix object.
            Regression design matrix. Each row corresponds to a distinct
            observation (trial), and each column to a distinct predictor
            (coefficient to fit). If <constant> == True, a constant
            (intercept) column will be appended to end of labels, if not already present.
            Number of rows(labels.shape[0] = n_obs) be same length as data.shape[axis].

    data    (...,n_obs,...) ndarray. Data to fit with regression model. <axis> should
            correspond to observations (trials), while rest of axis(s) are any
            independent data series (channels, time points, frequencies, etc.)
            that will be fit separately using the same list of group labels.

    axis    Int. Data axis corresponding to distinct observations/trials. Default: 0

    col_terms (n_params,) array-like. Lists regression term (eg as integer or string
            name) corresponding to each column (predictor) in labels. Mapping may not
            be 1:1 due to multiple dummy-variable columns arising from
            categorical terms with > 2 levels. PEV/stats are computed separately
            for all columns/predictors of each term pooled together.
            Default: If DesignMatrix, obtained from its attributes.
            Otherwise, assume 1:1 mapping from term:column (col_terms = np.arange(n_params))

    omega   Bool. If True, uses bias-corrected omega-squared formula for PEV,
            otherwise uses R-squared formula, which is positively biased.
            Default: True

    constant Bool. If True, ensures there is a constant column in labels to fit an
            intercept/bias term (appends if missing, does nothing if present).
            Default: True (include constant/intercept term)

    partial Bool. If True, uses partial-factor formula for PEV, where each term
            EV is expressed relative to only that term + error variance, and
            thus changes in one term's EV do not necessarily affect other terms.
            Otherwise, the standard full-model PEV formula is used. Default: False

    total   Bool. If True, appends total model explained variance (sum of all
            individual terms) to end of term axis. Default: False

    as_pct  Bool. If True, returns PEV as a percent (range ~ 0-100), else PEV
            is returned as a proportion (range ~ 0-1). Default: True

    return_stats Bool. If True, computes and returns several statistics of fitted
            model in addition to PEV. Default: False

    RETURNS
    exp_var (...,n_terms,...). Percent (or proportion) of variance in data explained by labels.
            Shape is same as data, with observation axis reduced to length = n_terms.

    stats   Dict. If <return_stats> set, statistics on each fit also returned:
        p   (...,n_terms,...). F-test p values for each datapoint. Same shape as exp_var.
        F   (...,n_terms,...). F-statistic for each datapoint. Same shape as exp_var.
        B   (...,nPredictors,...). Fitted regression coefficients for each predictor
            (column in labels). Same shape as exp_var, but with B.shape[axis] = nPredictors.

    REFERENCE
    Regression eqn's: Draper & Smith _Applied Regression Analysis (1998) sxn's 1.3, 6.1
    Omega^2 stat:     Snyder & Lawson (1993) J of Experimental Education
                      wikipedia.org/wiki/Effect_size
    """
    if not isinstance(labels,DesignMatrix): labels = np.asarray(labels)

    # Reshape data array data -> (n_observations,n_data_series) matrix
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    n_obs,n_series = data.shape

    assert labels.shape[0] == n_obs, \
            "Design matrix <labels> and data array should have same number of rows (%d != %d)" \
            % (labels.shape[0], n_obs)

    # If col_terms not set, obtain from DesignMatrix or assume each column is a distinct term
    if col_terms is None:
        if isinstance(labels,DesignMatrix): col_terms = patsy_terms_to_columns(labels)
        else:                               col_terms = np.arange(labels.shape[1])
    col_terms = np.asarray(col_terms)

    # If a constant is requested and not already present in design matrix <labels>,
    #  concatenate (n_obs,) vector of ones to end of design matrix
    constant_col = np.all(labels==1,axis=0)
    if constant and not np.any(constant_col):
        labels      = np.hstack((labels,np.ones((n_obs,1))))
        col_terms   = np.hstack((col_terms,np.nan))
        constant_col = np.hstack((constant_col,True))

    n_params    = labels.shape[1]

    term_set    = unsorted_unique(col_terms[~constant_col])
    n_terms     = len(term_set)

    grand_mean  = np.mean(data, axis=0)
    SS_total    = np.sum((data - grand_mean)**2, axis=0) # Total Sums of Squares

    # Create linear regression object
    model       = LinearRegression()

    # Fit full model to data, save coefficients, and compute prediction of data
    model.fit(labels,data)
    # Reshape coeffs (n_series,nPredictors) -> (nPredictors,n_series)
    B           = model.coef_.T
    # Full-model Error Sums of Squares
    SS_error_full= np.sum((data - model.predict(labels))**2, axis=0)

    # Fit reduced models eliminating each term, in turn, and calculate
    #  additional sums of squares for each (ie full model minus each factor)
    SS_extra = np.zeros((n_terms,n_series))
    df_extra = np.zeros((n_terms,))
    for i_term,term in enumerate(term_set):
        term_idxs = col_terms == term
        df_extra[i_term]    = np.sum(term_idxs)   # Regression degrees of freedom for term

        # Fit/predict with reduced model, without current-term <labels> columns
        labels_reduced      = labels[:,~term_idxs]
        model.fit(labels_reduced,data)
        # Reduced-model Error Sums of Squares
        SSerror_reduced     = np.sum((data - model.predict(labels_reduced))**2, axis=0)
        # Extra regression Sums-of-Squares for term
        SS_extra[i_term,:]  = SSerror_reduced - SS_error_full

    # PEV,F strictly undefined when no data variance (div by 0).
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

    exp_var = undo_standardize_array(exp_var, data_shape, axis=axis, target_axis=0)

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
        F[:,undefined] = 0                    # Set F = 0 for data w/ data variance = 0
        p           = Ftest.sf(F,df_extra,df_error) # p value for given F stat

        F   = undo_standardize_array(F, data_shape, axis=axis, target_axis=0)
        p   = undo_standardize_array(p, data_shape, axis=axis, target_axis=0)
        B   = undo_standardize_array(B, data_shape, axis=axis, target_axis=0)

        stats   = {'p':p, 'F':F, 'B':B}
        return exp_var, stats


# =============================================================================
# Design matrix-related functions
# =============================================================================
def patsy_terms_to_columns(labels):
    """
    Given a patsy DesignMatrix, maps model terms to design matrix columns,
    returning a vector listing the term corresponding to each column.

    Note that this correspondence may not be 1:1 due to categorical terms
    generating multiple dummy columns.

    col_terms = patsy_terms_to_columns(labels)

    ARGS
    labels   (n_observations,n_columns) patsy DesignMatrix object.

    RETURNS
    col_terms (n_columns,) array of strings. Lists term name (from labels.design_info.term_names)
            corresponding to each column in design matrix <labels>.
    """
    # todo  Should we option list of int indexes instead string names? faster downstream?
    assert isinstance(labels,DesignMatrix), \
        ValueError("patsy_terms_to_columns: <labels> must be a patsy DesignMatrix object")

    n_cols  = labels.shape[1]
    col_terms = np.full((n_cols,),fill_value='',dtype=object)

    # For each term in design matrix, find all columns it maps to from design_info
    # attribute in patsy DesignMatrix object, and insert term name string
    # into corresponding columns in output variable
    for term,slicer in labels.design_info.term_name_slices.items():
        col_terms[slicer] = term

    return col_terms


# =============================================================================
# Low-level information metric computation functions
# =============================================================================
def entropy(P):
    """ Computes entropy from probabilty density P """
    return -(P * np.log2(P)).sum()


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


def omega_squared_partial(SS_model, SS_total, MS_error, df_model, n_obs):
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
