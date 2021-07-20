# -*- coding: utf-8 -*-
"""
utils   A module of Python utilities helpful for data

FUNCTIONS
### Numerical utility functions ###
set_random_seed     Seeds Python/Nummpy random number generators with given seed
interp1             Interpolates 1d data vector at given index values
fano                Computes Fano factor (variance/mean) of data
cv                  Computes Coefficient of Variation (SD/mean) of data
cv2                 Computes local Coefficient of Variation (Holt 1996) of data
lv                  Computes Local Variation (Shinomoto 2009) of data
zscore              Z-scores data along given axis (or whole array)

### Data indexing and reshaping functions ###
index_axis          Dynamically index into arbitrary axis of ndarray
axis_index_slices   Generates list of slices for dynamic axis indexing
standardize_array   Reshapes array to 2D w/ "business" axis at 0 or -1 for analysis
undo_standardize_array Undoes effect of standardize_array after analysis

### Other utilities ###
iarange             np.arange(), but with an inclusive endpoint
unsorted_unique     np.unique(), but without sorting values
isarraylike         Tests if variable is "array-like" (ndarray, list, or tuple)
setup_sliding_windows   Generates set of sliding windows using given parameters

Created on Fri Apr  9 13:28:15 2021

@author: sbrincat
"""
import time
import random
import numpy as np

from scipy.interpolate import interp1d


# =============================================================================
# Numerical utility functions
# =============================================================================
def set_random_seed(seed=None):
    """
    Seeds built-in Python and Nummpy random number generators with given seed

    seed = set_random_seed(seed=None)

    INPUT
    seed    Int | String. Seed to use. If string given, converts each char to
            ascii and sums the resulting values. If no seed given, seeds based
            on current clock time

    OUTPUT
    seed    Int. Actual integer seed used
    """
    if seed is None:            seed = int(time.time()*1000.0) % (2**32 - 1)
    # Convert string seeds to int's (convert each char->ascii and sum them)
    elif isinstance(seed,str):  seed = np.sum([ord(c) for c in seed])

    # Set Numpy random number generator
    np.random.seed(seed)
    # Set built-in random number generator in Python random module
    random.seed(seed)

    return seed


def interp1(x, y, xinterp, **kwargs):
    """
    Interpolates 1d data vector <y> sampled at index values <x> to
    new sampling vector <xinterp>
    Convenience wrapper around scipy.interpolate.interp1d w/o weird call structure
    """
    return interp1d(x,y,**kwargs).__call__(xinterp)


def fano(data, axis=None, ddof=0):
    """
    Computes Fano factor (variance/mean) of data along a given array axis
    (eg trials) or across an entire array
    
    np.nan is returned for cases where the mean ~ 0
    
    Fano factor has an expected value of 1 for a Poisson distribution/process.
    
    fano = fano(data, axis=None, ddof=0)
    
    ARGS
    data        (...,n_obs,...) ndarray. Data of arbitrary shape.
    
    axis        Int. Array axis to compute Fano factor on (usually corresponding to
                distict trials/observations). If None [default], computes Fano
                factor across entire array (analogous to np.mean/var).
                
    ddof        Int. Sets divisor for computing variance = N - ddof. Set=0 for max 
                likelihood estimate, set=1 for unbiased (N-1 denominator) estimate.
                Default: 0
                                
    RETURNS
    fano        Float | (...,1,...) ndarray. Fano factor (variance/mean) of data.
                For vector data or axis=None, a single scalar value is returned.
                Otherwise, it's an array w/ same shape as data, but with <axis>
                reduced to length 1.   
    """
    mean    = data.mean(axis=axis,keepdims=True)
    var     = data.var(axis=axis,keepdims=True,ddof=ddof)
    fano_   = var/mean
    # Find any data values w/ mean ~ 0 and set output = NaN for those points    
    fano_[np.isclose(mean,0)] = np.nan

    if fano_.size == 1: fano_ = fano_.item()
    
    return fano_

# Alias fano() as fano_factor()
fano_factor = fano


def cv(data, axis=None, ddof=0):
    """
    Computes Coefficient of Variation (std dev/mean) of data, computed along
    a given array axis (eg trials) or across an entire array
    
    np.nan is returned for cases where the mean ~ 0
    
    CV has an expected value of 1 for a Poisson distribution/process.
    
    CV = cv(data, axis=None, ddof=0)
    
    ARGS
    data        (...,n_obs,...) ndarray. Data of arbitrary shape.
    
    axis        Int. Array axis to compute CV on (usually corresponding to
                distict trials/observations). If None [default], computes Fano
                factor across entire array (analogous to np.mean/var).
                
    ddof        Int. Sets divisor for computing std dev = N - ddof. Set=0 for max 
                likelihood estimate, set=1 for unbiased (N-1 denominator) estimate.
                Default: 0
                                
    RETURNS
    CV          Float | (...,1,...) ndarray. CV (SD/mean) of data.
                For vector data or axis=None, a single scalar value is returned.
                Otherwise, it's an array w/ same shape as data, but with <axis>
                reduced to length 1.   
    """    
    mean    = data.mean(axis=axis,keepdims=True)
    sd      = data.std(axis=axis,keepdims=True,ddof=ddof)
    CV      = sd/mean
    # Find any data values w/ mean ~ 0 and set output = NaN for those points    
    CV[np.isclose(mean,0)] = np.nan

    if CV.size == 1: CV = CV.item()
    
    return CV

# Alias cv() as coefficient_of_variation()
coefficient_of_variation = cv


def cv2(data, axis=0):
    """
    Computes local Coefficient of Variation (CV2) of data, computed along
    a given array axis (eg trials). Typically used as measure of local variation
    in inter-spike intervals.
    
    CV2 reduces effects of slow changes in data (eg changes in spike rate) on 
    measure of variation by only comparing adjacent data values (eg adjacent ISIs).
    CV2 has an expected value of 1 for a Poisson process.
        
    CV2 = cv2(data, axis=0)
    
    ARGS
    data        (...,n_obs,...) ndarray. Data of arbitrary shape.
    
    axis        Int. Array axis to compute CV2 on (usually corresponding to
                distict trials/observations). Default: 0
                                
    RETURNS
    CV2         Float | (...,1,...) ndarray. CV2 of data.
                For vector data or axis=None, a single scalar value is returned.
                Otherwise, it's an array w/ same shape as data, but with <axis>
                reduced to length 1.
                
    REFERENCE
    Holt et al. (1996) Journal of Neurophysiology https://doi.org/10.1152/jn.1996.75.5.1806
    """
    # Difference between adjacent values in array, along <axis>
    diff    = np.diff(data,axis=axis)
    # Sum between adjacent values in array, along <axis>
    denom   = index_axis(data, axis, range(0,data.shape[axis]-1)) + \
              index_axis(data, axis, range(1,data.shape[axis]))
    
    # CV2 formula (Holt 1996 eqn. 4)
    CV2     = (2*np.abs(diff) / denom).mean(axis=axis)
        
    if CV2.size == 1: CV2 = CV2.item()
    
    return CV2


def lv(data, axis=0):
    """
    Computes Local Variation (LV) of data, computed along a given array axis (eg trials).
    Typically used as measure of local variation in inter-spike intervals.
    
    LV reduces effects of slow changes in data (eg changes in spike rate) on 
    measure of variation by only comparing adjacent data values (eg adjacent ISIs).
    LV has an expected value of 1 for a Poisson process.
            
    LV = lv(data, axis=0)
    
    ARGS
    data        (...,n_obs,...) ndarray. Data of arbitrary shape.
    
    axis        Int. Array axis to compute LV on (usually corresponding to
                distict trials/observations). Default: 0
                                
    RETURNS
    LV          Float | (...,1,...) ndarray. LV of data.
                For vector data or axis=None, a single scalar value is returned.
                Otherwise, it's an array w/ same shape as data, but with <axis>
                reduced to length 1.
                
    REFERENCE
    Shinomoto et al. (2009) PLoS Computational Biology https://doi.org/10.1371/journal.pcbi.1000433
    """
    # Difference between adjacent values in array, along <axis>
    diff    = np.diff(data,axis=axis)
    # Sum between adjacent values in array, along <axis>
    denom   = index_axis(data, axis, range(0,data.shape[axis]-1)) + \
              index_axis(data, axis, range(1,data.shape[axis]))
    n       = data.shape[axis]
    
    # LV formula (Shinomoto 2009 eqn. 2)
    # Note: np.diff() reverses sign from original formula, but it gets squared anyway
    LV     = (((diff/denom)**2) * (3/(n-1))).sum(axis=axis)
        
    if LV.size == 1: LV = LV.item()
    
    return LV



# =============================================================================
# Pre/post-processing utility functions
# =============================================================================
def zscore(data, axis=None, time_range=None, time_axis=None, timepts=None,
           ddof=0, zerotol=1e-6, return_stats=False):
    """
    Z-scores data along given axis (or whole array)
    
    Optionally also returns mean,SD (eg, to compute on training set, apply to test set)    
    
    data = zscore(data, axis=None, time_range=None, time_axis=None, timepts=None,
                  ddof=0, zerotol=1e-6, return_stats=False)
                  
    data, mu, sd = zscore(data, axis=None, time_range=None, time_axis=None, timepts=None,
                          ddof=0, zerotol=1e-6, return_stats=True)
    
    ARGS
    data        Array-like (arbitrary dimensionality). Data to z-score.
    
    axis        Int. Array axis to compute mean/SD along for z-scoring (usually corresponding to
                distict trials/observations). If None [default], computes mean/SD across entire
                array (analogous to np.mean/std).
            
    time_range  (2,) array-like. Optionally allows for computing mean/SD within a given time
                window, then using these z-score ALL timepoints (eg compute mean/SD within
                a "baseline" window, then use to z-score all timepoints). Set=[start,end] 
                of time window. If set, MUST also provide values for time_axis and timepts.
                Default: None (compute mean/SD over all time points)
            
    time_axis   Int. Axis corresponding to timepoints. Only necessary if time_range is set.
    
    timepts     (n_timepts,) array-like. Time sampling vector for data. Only necessary if
                time_range is set, unused otherwise.
                
    ddof        Int. Sets divisor for computing SD = N - ddof. Set=0 for max likelihood estimate,
                set=1 for unbiased (N-1 denominator) estimate. Default: 0

    zerotol     Float. Any SD values < zerotol are treated as 0, and corresponding z-scores 
                set = nan. Default: 1e-6
                
    return_stats Bool. If True, also returns computed mean, SD. If False [default], only returns
                z-scored data.                
        
    RETURNS
    data        Z-scored data. Same shape as input data.
    
    - optional outputs only returned if return_stats=True    
    mean        Computed means for z-score
    sd          Computed standard deviations for z-score
                Mean and sd both have same shape as data, but with <axis> reduced to length 1.
    """
    # Compute mean/SD separately for each timepoint (axis not None) or across all array (axis=None)
    if time_range is None:
        # Compute mean and standard deviation of data along <axis> (or entire array)
        mu = data.mean(axis=axis, keepdims=True)        
        sd = data.std(axis=axis, ddof=ddof, keepdims=True)
        
    # Compute mean/SD within given time range, then apply to all timepoints        
    else:
        assert (len(time_range) == 2) and (time_range[1] > time_range[0]), \
            "time_range must be given as [start,end] time of desired time window"
        assert timepts is not None, "If time_range is set, must also input value for timepts"
        assert time_axis is not None, "If time_range is set, must also input value for time_axis"
        
        # Compute mean data value across all timepoints within window = "baseline" for z-score
        win_bool = (timepts >= time_range[0]) & (timepts <= time_range[1])
        win_data = index_axis(data, time_axis, win_bool).mean(axis=time_axis, keepdims=True)
        
        # Compute mean and standard deviation of data along <axis>       
        mu = win_data.mean(axis=axis, keepdims=True)        
        sd = win_data.std(axis=axis, ddof=ddof, keepdims=True)
    
    # Compute z-score -- Subtract mean and normalize by SD
    data = (data - mu) / sd

    # Find any data values w/ sd ~ 0 and set data = NaN for those points
    zero_points = np.isclose(sd,0,rtol=zerotol)
    tiling = [1]*data.ndim
    tiling[axis] = data.shape[axis]
    if time_range is not None: tiling[time_axis] = data.shape[time_axis]
    data[np.tile(zero_points,tiling)] = np.nan
        
    if return_stats:    return data, mu, sd
    else:               return data
    
    
# =============================================================================
# Data indexing and reshaping functions
# =============================================================================
def index_axis(data, axis, idxs):
    """ 
    Utility to dynamically index into a arbitrary axis of an ndarray 
    
    Similar to function of Numpy take and compress functions, but this can take either
    integer indexes, boolean indexes, or a slice object. And this is generally much faster.
    
    data = index_axis(data, axis, idxs)
    
    ARGS
    data    ndarray. Array of arbitrary shape, to index into given axis of.
    
    axis    Int. Axis of ndarray to index into.
    
    idxs    (n_selected,) array-like of int | (axis_len,) array-like of bool | slice object
            Indexing into given axis of array, given either as list of
            integer indexes or as boolean vector.
    
    RETURNS
    data    ndarray. Input array with indexed values selected from given axis.    
    """
    # Generate list of slices, with ':' for all axes except <idxs> for <axis>
    slices = axis_index_slices(axis, idxs, data.ndim)

    # Use slices to index into data, and return sliced data
    return data[slices]


def axis_index_slices(axis, idxs, ndim):
    """
    Generate list of slices, with ':' for all axes except <idxs> for <axis>,
    to use for dynamic indexing into an arbitary axis of an ndarray
    
    slices = axis_index_slices(axis, idxs, ndim)
    
    ARGS
    axis    Int. Axis of ndarray to index into.
    
    idxs    (n_selected,) array-like of int | (axis_len,) array-like of bool | slice object
            Indexing into given axis of array, given either as list of
            integer indexes or as boolean vector.
    
    ndim    Int. Number of dimensions in ndarray to index into
    
    RETURNS
    slices  Tuple of slices. Index tuple to use to index into given 
            axis of ndarray as: selected_values = array[slices]  
    """
    # Initialize list of null slices, equivalent to [:,:,:,...]
    slices = [slice(None)] * ndim

    # Set slice for <axis> to desired indexing
    slices[axis] = idxs

    # Convert to tuple bc indexing arrays w/ a list is deprecated
    return tuple(slices)


def standardize_array(data, axis=0, target_axis=0):
    """
    Reshapes multi-dimensional data array to standardized 2D array (matrix-like) form, 
    with "business" axis shifted to axis = <target_axis> for analysis

    data, data_shape = standardize_array(data,axis=0,target_axis=0)

    ARGS
    data        (...,n,...) ndarray. Data array of arbitrary shape.
    
    axis        Int. Axis of data to move to <target_axis> for subsequent analysis
                Default: 0
                
    target_axis Int. Array axis to move <axis> to for subsequent analysis
                MUST be 0 (first axis) or -1 (last axis). Default: 0

    RETURNS
    data    (n,m) | (m,n) ndarray. Data array w/ <axis> moved to <target_axis>, 
            and all other axes unwrapped into single dimension,  
            where m = prod(shape[axes != axis])
            
    data_shape (data.ndim,) tuple. Original shape of data array

    Note:   Even 1d (vector) data is expanded into 2d (n,1) | (1,n) array to
            standardize for calling code.
    """
    assert target_axis in [0,-1], \
        ValueError("target_axis set = %d. Must be 0 (first axis) or -1 (last axis)" % target_axis)
        
    if target_axis == 0:    return standardize_to_axis_0(data, axis=axis)
    else:                   return standardize_to_axis_end(data, axis=axis)                
    
    
def undo_standardize_array(data, data_shape, axis=0, target_axis=0):
    """
    Undoes effect of standardize_array() -- reshapes data array from unwrapped 
    2D (matrix-like) form back to ~ original multi-dimensional form, with <axis>
    shifted back to original location (but allowing that data.shape[axis] may have changed)

    data = undo_standardize_array(data,data_shape,axis=0,target_axis=0)

    ARGS
    data        (axis_len,m) | (m,axis_len) ndarray. Data array w/ <axis> moved to <target_axis>, 
                and all axes != <target_axis> unwrapped into single dimension, where 
                m = prod(shape[axes != axis])
            
    data_shape  (data.ndim,) tuple. Original shape of data array. 
                Second output of standardize_array.

    axis        Int. Axis of original data moved to <target_axis>, which will be shifted 
                back to original axis. Default: 0

    target_axis Int. Array axis <axis> was moved to for subsequent analysis
                MUST be 0 (first axis) or -1 (last axis). Default: 0

    RETURNS
    data        (...,axis_len,...) ndarray. Data array reshaped back to original shape
    """
    assert target_axis in [0,-1], \
        ValueError("target_axis set = %d. Must be 0 (first axis) or -1 (last axis)" % target_axis)
        
    if target_axis == 0:    return undo_standardize_to_axis_0(data, data_shape, axis=axis)
    else:                   return undo_standardize_to_axis_end(data, data_shape, axis=axis)                
        
        
def standardize_to_axis_0(data, axis=0):
    """
    Reshapes multi-dimensional data array to standardized 2D array (matrix-like) form, 
    with "business" axis shifted to axis 0 for analysis

    data, data_shape = standardize_to_axis_0(data,axis=0)

    ARGS
    data    (...,n,...) ndarray. Data array of arbitrary shape.
    
    axis    Int. Axis of data to move to axis 0 for subsequent analysis. Default: 0

    RETURNS
    data    (n,m) ndarray. Data array w/ <axis> moved to axis=0, 
            and all other axes unwrapped into single dimension,  
            where m = prod(shape[axes != axis])
            
    data_shape (data.ndim,) tuple. Original shape of data array

    Note:   Even 1d (vector) data is expanded into 2d (n,1) array to
            standardize for calling code.
    """
    if axis < 0: axis = data.ndim + axis 
    data = np.asarray(data)

    # Save original shape/dimensionality of <data>
    data_ndim  = data.ndim
    data_shape = data.shape
        
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

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix, then transpose
        if data_ndim > 2:   data = np.reshape(data,(-1,data_shape[axis]),order='C').T
        else:               data = data.T

    # Expand (n,) vector data to (n,1) to simplify downstream code
    if data_ndim == 1:  data = data[:,np.newaxis]

    return data, data_shape


def undo_standardize_to_axis_0(data, data_shape, axis=0):
    """
    Undoes effect of standardize_to_axis_0() -- reshapes data array from unwrapped 
    2D (matrix-like) form back to ~ original multi-dimensional form, with <axis>
    shifted back to original location (but allowing that data.shape[axis] may have changed)

    data = undo_standardize_to_axis_0(data,data_shape,axis=0)

    ARGS
    data    (axis_len,m) ndarray. Data array w/ <axis> moved to axis=0, 
            and all axes != 0 unwrapped into single dimension, where 
            m = prod(shape[1:])
            
    data_shape (data.ndim,) tuple. Original shape of data array. 
            Second output of standardize_to_axis_0.

    axis    Int. Axis of original data moved to axis 0, which will be shifted 
            back to original axis. Default: 0

    RETURNS
    data    (...,axis_len,...) ndarray. Data array reshaped back to original shape
    """
    if axis < 0: axis = data.ndim + axis    
    
    data_shape  = np.asarray(data_shape)
    data_ndim = len(data_shape) # Number of dimensions in original data
    axis_len  = data.shape[0]   # Length of dim 0 (will become dim <axis> again)

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    # (but with length of dimension <axis> = <axisLength>)
    if data_ndim > 2:
        # Reshape data -> (axis_len,<original shape w/o <axis>>)
        shape = (axis_len, *data_shape[np.arange(data_ndim) != axis])
        # Note: I think you want the order to be 'F' regardless of memory layout
        # TODO test this!!!        
        data = np.reshape(data,shape,order='F')

    # Squeeze (n,1) array back down to 1d (n,) vector, 
    #  and extract value from scalar array -> float
    elif data_ndim == 1:
        data = data.squeeze(axis=-1)
        if data.size == 1: data = data.item()

    # If observation axis wasn't 0, permute axis back to original position
    if (axis != 0) and isinstance(data,np.ndarray):
        data = np.moveaxis(data,0,axis)
    
    return data


def standardize_to_axis_end(data, axis=-1):
    """
    Reshapes multi-dimensional data array to standardized 2D array (matrix-like) form, 
    with "business" axis shifted to axis -1 (end) for analysis
    
    data, data_shape = standardize_to_axis_end(data,axis=-1)

    ARGS
    data    (...,n,...) ndarray. Data array of arbitrary shape.

    axis    Int. Axis of data to move to axis -1 for subsequent analysis. Default: -1

    RETURNS
    data    (m,n) ndarray. Data array w/ <axis> moved to axis=-1, 
            and all other axes unwrapped into single dimension,  
            where m = prod(shape[axes != axis])

    data_shape (data.ndim,) tuple. Original shape of data array

    Note:   Even 1d (vector) data is expanded into 2d (1,n) array to
            standardize for calling code.    
    """
    if axis < 0: axis = data.ndim + axis 
    data = np.asarray(data)
    
    # Save original shape/dimensionality of <data>
    data_ndim  = data.ndim
    data_shape = data.shape

    # Faster method for f-contiguous arrays
    if data.flags.f_contiguous:
        # If observation axis != first dim, permute axis to make it so
        if axis != 0: data = np.moveaxis(data,axis,0)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix, then transpose
        if data_ndim > 2:   data = np.reshape(data,(data_shape[axis],-1),order='F').T
        else:               data = data.T

    else:
        # If observation axis != -1, permute axis to make it so
        if axis != data_ndim - 1: data = np.moveaxis(data,axis,-1)

        # If data array data has > 2 dims, keep axis -1 and unwrap other dims into a matrix
        if data_ndim > 2:   data = np.reshape(data,(-1,data_shape[axis]),order='C')

    # Expand (n,) vector data to (1,n) to simplify downstream code
    if data_ndim == 1:  data = data[np.newaxis,:]

    return data, data_shape


def undo_standardize_to_axis_end(data, data_shape, axis=-1):
    """
    Undoes effect of standardize_to_axis_end() -- reshapes data array from unwrapped 
    2D (matrix-like) form back to ~ original multi-dimensional form, with <axis>
    shifted back to original location (but allowing that data.shape[axis] may have changed)
    
    data = undo_standardize_to_axis_end(data,data_shape,axis=-1)

    ARGS
    data    (m,axis_len) ndarray. Data array w/ <axis> moved to axis=-1, 
            and all axes != -1 unwrapped into single dimension, where 
            m = prod(shape[:-1])

    data_shape (data.ndim,) tuple. Original shape of data array
            Second output of standardize_to_axis_end.

    axis    Int. Axis of original data moved to axis -1, which will be shifted 
            back to original axis.. Default: -1

    RETURNS
    data    (...,axis_len,...) ndarray. Data array reshaped back to original shape
    """
    data_shape  = np.asarray(data_shape)

    data_ndim   = len(data_shape) # Number of dimensions in original data
    axis_len    = data.shape[-1]  # Length of dim -1 (will become dim <axis> again)

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    # (but with length of dimension <axis> = <axis_length>)
    if data_ndim > 2:
        # Reshape data -> (<original shape w/o <axis>>,axis_len)
        shape = (*data_shape[np.arange(data_ndim) != axis], axis_len)
        # Note: I think you want the order to be 'C' regardless of memory layout
        # TODO test this!!!
        data  = np.reshape(data,shape,order='C')

    # Squeeze (1,n) array back down to 1d (n,) vector, 
    #  and extract value from scalar array -> float
    elif data_ndim == 1:
        data = data.squeeze(axis=0)        
        if data.size == 1: data = data.item()
        
    # If observation axis wasn't -1, permute axis back to original position
    if (axis != -1) and isinstance(data,np.ndarray):
        data = np.moveaxis(data,-1,axis)

    return data


def concatenate_object_array(data,axis=None,sort=False):
    """
    Concatenates objects across one or more axes of an object array.
    Useful for concatenating spike timestamps across trials, units, etc.
    
    data = concatenate_object_array(data,axis=None,sort=False)
    
    EXAMPLE
    data = [[[1, 2],    [3, 4, 5]], 
            [[6, 7, 8], [9, 10]  ]]
    concatenate_object_array(data,axis=0)
    >> [[1,2,6,7,8], [3,4,5,9,10]]
    concatenate_object_array(data,axis=1)
    >> [[1,2,3,4,5], [6,7,8,9,10]]
    
    ARGS
    data    Object ndarray of arbitary shape containing 1d lists/arrays
    
    axis    Int | list of int | None. Axis(s) to concatenate object array across.
            Set = list of ints to concatenate across multiple axes.
            Set = None [default] to concatenate across *all* axes in data.

    sort    Bool. If True, sorts items in concatenated list objects. Default: False

    OUTPUTS
    data    Concatenated object(s).
            If axis is None, returns as single list extracted from object array.
            Otherwise, returns as object ndarray with all concatenated axes 
            reduced to singletons.
    """
    assert data.dtype == object, \
        ValueError("data is not an object array. Use np.concatenate() instead.")
    
    # Convert axis=None to list of *all* axes in data
    if axis is None:    axis_ = 0 if data.ndim == 1 else list(range(data.ndim))
    else:               axis_ = axis

    # If <axis> is a list of multiple axes, iterate thru each axis,
    # recursively calling this function on each axis in list
    if not np.isscalar(axis_):
        for ax in axis_:
            # Only need to sort for final concatenation axis
            sort_ = sort if ax == axis_[-1] else False
            data = concatenate_object_array(data,axis=ax,sort=sort_)
            
        # Extract single object if we concatenated across all axes (axis=None)            
        if axis is None: data = data.item()
        
        return data
    
    # If concatenation axis is already a singleton, we are done, return as-is
    if data.shape[axis_] == 1: return data    
                
                
    # Reshape data so concatenation axis is axis 0, all other axes are unwrapped to 2d array
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    n_series = data.shape[1]
    
    data_concat = np.empty((1,n_series),dtype=object)
    
    for j in range(n_series):
        # Concatenate objects across all entries 
        data_concat[0,j] = np.concatenate([values for values in data[:,j]])
        # Sort items in concatenated object, if requested
        if sort: data_concat[0,j].sort()
    
    # Reshape data back to original axis order and shape
    data_concat = undo_standardize_array(data_concat, data_shape, axis=axis, target_axis=0)

    return data_concat

    
# =============================================================================
# Other utility functions
# =============================================================================
def iarange(start=0, stop=0, step=1):
    """
    Implements Numpy arange() with an inclusive endpoint. Same inputs as arange(), same
    output, except ends at stop, not stop - 1 (or more generally stop - step)

    r = iarange(start=0,stop=0,step=1)
    
    Note: Must input all 3 arguments or use keywords (unlike flexible arg's in arange)    
    """
    # Offset to get final value in sequence is 1 for int-valued step, small float otherwise
    offset = 1 if isinstance(step,int) else 1e-12
    # Make offset negative for a negative step
    if step < 0: offset = -offset
    
    return np.arange(start,stop+offset,step)


def unsorted_unique(x, **kwargs):
    """
    Implements np.unique(x) without sorting, ie maintains original order of unique
    elements as they are found in x.

    SOURCE  stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    """
    x    = np.asarray(x)
    if 'axis' in kwargs:
        idxs = np.unique(x, return_index=True, **kwargs)[1]
        return index_axis(x, kwargs['axis'], np.sort(idxs))
    else:
        x = x.flatten()
        idxs = np.unique(x, return_index=True, **kwargs)[1]        
        return x[np.sort(idxs)]
        

def isarraylike(x):
    """
    Tests if variable <x> is "array-like": np.ndarray, list, or tuple
    Returns True if x is array-like, False otherwise
    """
    return isinstance(x, (list, tuple, np.ndarray))


def setup_sliding_windows(width, lims, step=None, reference=None,
                          force_int=False, exclude_end=None):
    """
    Generates set of sliding windows using given parameters

    windows = setup_sliding_windows(width,lims,step=None,
                                  reference=None,force_int=False,exclude_end=None)

    ARGS
    width       Scalar. Full width of each window. Required arg.

    lims        (2,) array-like. [start end] of full range of domain you want
                windows to sample. Required.

    step        Scalar. Spacing between start of adjacent windows
                Default: step = width (ie, perfectly non-overlapping windows)

    reference   Bool. Optionally sets a reference value at which one window
                starts and the rest of windows will be determined from there.
                eg, set = 0 to have a window start at x=0, or
                    set = -width/2 to have a window centered at x=0
                Default: None = just start at lim[0]

    force_int   Bool. If True, rounds window starts,ends to integer values.
                Default: False (don't round)

    exclude_end Bool. If True, excludes the endpoint of each (integer-valued)
                sliding win from the definition of that win, to prevent double-sampling
                (eg, the range for a 100 ms window is [1 99], not [1 100])
                Default: True if force_int==True, otherwise default=False

    OUTPUT
    windows     (n_wins,2) ndarray. Sequence of sliding window [start end]
    """
    # Default: step is same as window width (ie windows perfectly disjoint)
    if step is None: step = width
    # Default: Excluding win endpoint is default for integer-valued win's,
    #  but not for continuous wins
    if exclude_end is None:  exclude_end = True if force_int else False

    if exclude_end:
        # Determine if window params (and thus windows) are integer or float-valued
        params = np.concatenate((lims,width,step))
        is_int = np.allclose(np.round(params), params)
        # Set window-end offset appropriately--1 for int, otherwise small float value
        offset = 1 if is_int else 1e-12
        
    # Standard sliding window generation
    if reference is None:
        if exclude_end: win_starts = iarange(lims[0], lims[-1]-width+offset, step)
        else:           win_starts = iarange(lims[0], lims[-1]-width, step)

    # Origin-anchored sliding window generation
    #  One window set to start at given 'reference', position of rest of windows
    #  is set around that window
    else:
        if exclude_end:
            # Series of windows going backwards from ref point (flipped to proper order),
            # followed by Series of windows going forwards from ref point
            win_starts = np.concatenate((np.flip(iarange(reference, lims[0], -step)),
                                         iarange(reference+step, lims[-1]-width+offset, step)))

        else:
            win_starts = np.concatenate((np.flip(iarange(reference, lims[0], -step)),
                                         iarange(reference+step, lims[-1]-width, step)))

    # Set end of each window
    if exclude_end: win_ends = win_starts + width - offset
    else:           win_ends = win_starts + width

    # Round window starts,ends to nearest integer
    if force_int:
        win_starts = np.round(win_starts)
        win_ends   = np.round(win_ends)

    return np.stack((win_starts,win_ends),axis=1)
