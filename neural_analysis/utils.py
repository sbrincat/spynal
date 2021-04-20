# -*- coding: utf-8 -*-
"""
utils   A module of Python utilities helpful for data

FUNCTIONS
### Numerical utility functions ###
set_random_seed     Seeds Python/Nummpy random number generators with given seed
interp1             Interpolates 1d data vector at given index values

### Data indexing and reshaping functions ###
index_axis          Dynamically index into arbitrary axis of ndarray
axis_index_slices   Generates list of slices for dynamic axis indexing
standardize_array   Reshapes array to 2D w/ "business" axis at 0 or -1 for analysis
undo_standardize_array Undoes effect of standardize_array after analysis

### Other utilities ###
iarange             np.arange(), but with an inclusive endpoint
unsorted_unique     np.unique(), but without sorting values
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
    if isinstance(step,int):    return np.arange(start,stop+1,step)
    else:                       return np.arange(start,stop+1e-12,step)


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

    # Standard sliding window generation
    if reference is None:
        if exclude_end: win_starts = iarange(lims[0], lims[-1]-width+1, step)
        else:           win_starts = iarange(lims[0], lims[-1]-width, step)

    # Origin-anchored sliding window generation
    #  One window set to start at given 'reference', position of rest of windows
    #  is set around that window
    else:
        if exclude_end:
            # Series of windows going backwards from ref point (flipped to proper order),
            # followed by Series of windows going forwards from ref point
            win_starts = np.concatenate(np.flip(iarange(reference, lims[0], -step)),
                                        iarange(reference+step, lims[-1]-width+1, step))

        else:
            win_starts = np.concatenate(np.flip(iarange(reference, lims[0], -step)),
                                        iarange(reference+step, lims[-1]-width, step))

    # Set end of each window
    if exclude_end: win_ends = win_starts + width - 1
    else:           win_ends = win_starts + width

    # Round window starts,ends to nearest integer
    if force_int:
        win_starts = np.round(win_starts)
        win_ends   = np.round(win_ends)

    return np.stack((win_starts,win_ends),axis=1)
