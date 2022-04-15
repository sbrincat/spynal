# -*- coding: utf-8 -*-
"""
helpers   Private helper functions for neural_analysis code

These are functions used internally in one or more analysis modules,
but not intended to be public-facing.

Created on Fri Apr  9 14:08:15 2021

@author: sbrincat
"""
import numpy as np

from scipy.stats import mode


def _check_window_lengths(windows,tol=1):
    """
    Ensures a set of windows are the same length. If not equal, but within given tolerance,
    windows are trimmed or expanded to the modal window length.

    ARGS
    windows (n_wins,2) array-like. Set of windows to test, given as series of [start,end].

    tol     Scalar. Max tolerance of difference of each window length from the modal value.

    RETURNS
    windows (n_wins,2) ndarray. Same windows, possibly slightly trimmed/expanded to uniform length
    """
    windows = np.asarray(windows)

    window_lengths  = np.diff(windows,axis=1).squeeze()
    window_range    = np.ptp(window_lengths)

    # If all window lengths are the same, windows are OK and we are done here
    if np.allclose(window_lengths, window_lengths[0]): return windows

    # Compute mode of windows lengths and max difference from it
    modal_length    = mode(window_lengths)[0][0]
    max_diff        = np.max(np.abs(window_lengths - modal_length))

    # If range is beyond our allowed tolerance, throw an error
    assert max_diff <= tol, \
        ValueError("All windows must have same length (input range=%.1f)" % window_range)

    # If range is between 0 and tolerance, we trim/expand windows to the modal length
    windows[:,1]    = windows[:,1] + (modal_length - window_lengths)
    return windows


def _enclose_in_object_array(data):
    """ Enclose array within an object array """
    out = np.empty((1,),dtype=object)
    out[0] = data
    return out


def _isbinary(x):
    """ Test whether variable contains only binary values (True,False,0,1) """
    x = np.asarray(x)
    return (x.dtype == bool) or \
           (np.issubdtype(x.dtype,np.number) and \
            np.all(np.in1d(x,[0,0.0,1,1.0,True,False])))


def _has_method(obj, method):
    """
    Determines if given object class instance has given method

    INPUTS
    obj     Object class instance (of any type) to test for method
    method  String. Name of method to test for

    RETURNS
    tf      Bool. True if obj.method exists; False otherwise

    SOURCE
    stackoverflow.com/questions/7580532/how-to-check-whether-a-method-exists-in-python/7580687
    """
    return callable(getattr(obj, method, None))


def _merge_dicts(dict1, dict2):
    """ Merges two dictionaries, with values in dict2 overriding (default) values in dict1 """
    dict_out = dict1.copy()
    dict_out.update(dict2)
    return dict_out


def _standardize_to_axis_0(data, axis=0):
    """
    Reshapes multi-dimensional data array to standardized 2D array (matrix-like) form,
    with "business" axis shifted to axis 0 for analysis

    data, data_shape = _standardize_to_axis_0(data,axis=0)

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

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims
        # into a matrix, then transpose
        if data_ndim > 2:   data = np.reshape(data,(-1,data_shape[axis]),order='C').T
        else:               data = data.T

    # Expand (n,) vector data to (n,1) to simplify downstream code
    if data_ndim == 1:  data = data[:,np.newaxis]

    return data, data_shape


def _undo_standardize_to_axis_0(data, data_shape, axis=0):
    """
    Undoes effect of _standardize_to_axis_0() -- reshapes data array from unwrapped
    2D (matrix-like) form back to ~ original multi-dimensional form, with <axis>
    shifted back to original location (but allowing that data.shape[axis] may have changed)

    data = _undo_standardize_to_axis_0(data,data_shape,axis=0)

    ARGS
    data    (axis_len,m) ndarray. Data array w/ <axis> moved to axis=0,
            and all axes != 0 unwrapped into single dimension, where
            m = prod(shape[1:])

    data_shape (data.ndim,) tuple. Original shape of data array.
            Second output of _standardize_to_axis_0.

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


def _standardize_to_axis_end(data, axis=-1):
    """
    Reshapes multi-dimensional data array to standardized 2D array (matrix-like) form,
    with "business" axis shifted to axis -1 (end) for analysis

    data, data_shape = _standardize_to_axis_end(data,axis=-1)

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

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims
        # into a matrix, then transpose
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


def _undo_standardize_to_axis_end(data, data_shape, axis=-1):
    """
    Undoes effect of _standardize_to_axis_end() -- reshapes data array from unwrapped
    2D (matrix-like) form back to ~ original multi-dimensional form, with <axis>
    shifted back to original location (but allowing that data.shape[axis] may have changed)

    data = _undo_standardize_to_axis_end(data,data_shape,axis=-1)

    ARGS
    data    (m,axis_len) ndarray. Data array w/ <axis> moved to axis=-1,
            and all axes != -1 unwrapped into single dimension, where
            m = prod(shape[:-1])

    data_shape (data.ndim,) tuple. Original shape of data array
            Second output of _standardize_to_axis_end.

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
