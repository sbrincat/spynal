#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for loading from and saving to Matlab v7 (and earlier) MAT files using scipy.io
"""
import numpy as np

import scipy.io

from spynal.helpers import _isbinary
from spynal.matIO.matIO import _parse_typemap, _is_structured_array, \
                               _structuredarray_to_dict, _structuredarray_to_dataframe, DEBUG


def _load7(filename, variables=None, typemap=None, order='Matlab'):
    """
    Loads data variables from a version 7 (or older) Matlab MAT file
    Uses scipy.io.loadmat to load data
    """
    typemap = _parse_typemap(typemap)

    # scipy.io returns arrays in column-major (Matlab/Fortran) order by default
    # Transpose/permute array axes if Python/C/row-major requested
    transpose = order.upper() in ['PYTHON','C','ROW','ROWMAJOR']

    # If <variables> is set, convert to {variable_name:None} dict
    # (otherwise leave = None, so it loads all variables from datafile)
    if variables is not None:
        variables = {vbl:None for vbl in variables}

    # Load each requested variable and save each as Numpy array in a {variable_name:array} dict
    data = scipy.io.loadmat(filename,variable_names=variables)

    # Get rid of header info returned by loadmat() (variables starting with '_...')
    data = {vbl:value for vbl,value in data.items() if vbl[0] != '_'}

    # Ensure we actually loaded all requested variables
    if variables is not None:
        for vbl in variables:
            assert vbl in data.keys(), \
                ValueError("Variable '%s' not present in file %s" % (vbl,filename))

    # If <variables> not set, load all variables in datafile
    else:
        variables = list(data.keys())

    # Get data into desired output format
    for vbl in variables:
        # Infer Matlab variable type from object structure
        matlab_vbl_type = _v7_matlab_type(data[vbl])
        # If specific variable name is listed in <typemap>, use given mapping
        python_vbl_type = typemap[vbl] if vbl in typemap else None

        if DEBUG: print("'%s'" % vbl, matlab_vbl_type, python_vbl_type)

        # Process loaded object -- extract data, convert to appropriate Python type,
        # traversing down into object (cell elements/struct fields) with recursive calls as needed
        data[vbl] = _process_v7_object(data[vbl], matlab_vbl_type=matlab_vbl_type,
                                       python_vbl_type=python_vbl_type, typemap=typemap,
                                       transpose=transpose, level=0)

    return data


def _who7(filename, **kwargs):
    """  Lists data variables from a version 7 (or older) Matlab MAT file """
    # Load list of 3-tuples of (variable,size,type) for each variable in file
    variables = scipy.io.whosmat(filename, appendmat=True, **kwargs)
    # Extract and return just the variable names
    return [vbl[0] for vbl in variables]


def _save7(filename, variables, **kwargs):
    """
    Saves data variables to version 7 Matlab MAT file
    Uses scipy.io.savemat to load data
    """
    scipy.io.savemat(filename,variables,**kwargs)


# =============================================================================
# Helper functions
# =============================================================================
# NOTE: This function can't be nested in _load7 b/c _dict_to_dataframe() needs access
def _process_v7_object(obj, matlab_vbl_type=None, python_vbl_type=None,
                       typemap=None, transpose=False, level=0):
    """ Properly handles arbitrary objects loaded from v7 mat files in recursive fashion """
    level += 1

    # Matlab class objects are returned as weird objects that are not parsable. Return as None.
    if matlab_vbl_type == 'class':
        converted = None

    # Matlab scalar structs are returned as (1,1) Numpy structured arrays
    #  Convert to dict or Pandas DataFrame (or leave as structured array)
    elif matlab_vbl_type == 'struct':
        if DEBUG: print('\t'*level, "struct")
        if python_vbl_type is None: python_vbl_type = typemap['struct']

        if python_vbl_type.lower() == 'dict':
            converted = _structuredarray_to_dict(obj[0,0], typemap=typemap,
                                                 transpose=transpose, level=level)

        elif python_vbl_type.lower() == 'dataframe':
            converted = _structuredarray_to_dataframe(obj[0,0], typemap=typemap,
                                                      transpose=transpose, level=level)

        else:
            converted = obj[0,0]
            assert python_vbl_type.lower() != 'structuredarray', \
                ValueError("%s is an unsupported output type for Matlab structs. \n \
                            Must be 'dict'|'dataFrame'|'structuredArray'" % python_vbl_type)

    # Matlab struct arrays are returned as (m,n) Numpy structured arrays
    #  Convert to dict (or leave as structured array)
    elif matlab_vbl_type == 'structarray':
        if DEBUG: print('\t'*level, "structarray")
        if python_vbl_type is None: python_vbl_type = typemap['structarray']

        if python_vbl_type.lower() == 'dict':
            converted = _structuredarray_to_dict(obj, typemap=typemap,
                                                 transpose=transpose, level=level)

        else:
            converted = obj
            assert python_vbl_type.lower() != 'structuredarray', \
                ValueError("%s is an unsupported output type for Matlab struct arrays. \n \
                            Must be 'dict'|'structuredArray'" % python_vbl_type)

    # Cell arrays are returned as object arrays. Squeeze out any singleton axes and
    # Permute axes of each cell's contents if 'PYTHON' order requested
    elif matlab_vbl_type == 'cellarray':
        if DEBUG: print('\t'*level, "cellarray")
        flatiter = obj.flat             # 1D iterator to iterate over arbitrary-shape array
        converted = np.ndarray(shape=obj.shape,dtype=object) # Create empty object array
        for _ in range(obj.size):
            coords = flatiter.coords    # Multidim coordinates into array
            # Infer Matlab variable type of cell array element
            matlab_elem_type = _v7_matlab_type(obj[coords])
            converted[coords] = _process_v7_object(obj[coords], typemap=typemap,
                                                   matlab_vbl_type=matlab_elem_type,
                                                   transpose=transpose, level=level)
            next(flatiter)              # Iterate to next element

    # General numerical/logical and cell arrays
    elif isinstance(obj,np.ndarray):
        if DEBUG: print('\t'*level, "array")
        # Logicals (binary-valued variables): Convert to bool type
        if matlab_vbl_type == 'logical':
            if DEBUG: print('\t'*level, "logical")
            converted = obj.astype(bool)

        # Strings: Extract string (dealing with empty strings appropriately)
        elif obj.dtype.type == np.str_:
            if DEBUG: print('\t'*level, "string")
            # Empty strings
            if len(obj) == 0:   converted = str()
            # General strings: Convert to string type (from np.str_)
            else:               converted = str(obj[0])

        # General numerical array: Copy as-is
        else:
            if DEBUG: print('\t'*level, "numerical")
            converted = obj

    # Note: Only do the following for objects that remain output as ndarrays
    if isinstance(converted,np.ndarray):
        # Length-1 arrays: Extract single item as type <dtype>
        if (converted.shape == ()) or (converted.size == 1):
            converted = converted.item()

        else:
            # Squeeze out any singleton axes, eg: Reshape (1,n) ndarrays -> (n,) vectors
            converted = converted.squeeze()

            # Permute array axes if 'PYTHON'/'C'/'ROWMAJOR' order requested
            if transpose: converted = converted.T

    return converted


def _v7_matlab_type(obj):
    """ Infer Matlab variable type of objects loaded from v7 mat files """
    # Matlab class objects are returned as this weird object (which is unparsable)
    if isinstance(obj,scipy.io.matlab.mio5_params.MatlabOpaque):    return 'class'
    # Matlab scalar structs are returned as (1,1) Numpy structured arrays
    elif _is_structured_array(obj) and (obj.size == 1):             return 'struct'
    # Matlab struct arrays are returned as (m,n) Numpy structured arrays
    elif _is_structured_array(obj) and (obj.size > 1):              return 'structarray'
    elif isinstance(obj,np.ndarray):
        # Cell arrays are returned as object arrays
        if obj.dtype == object:                                     return 'cellarray'
        # Heuristically assume arrays with only 0/1 (but not all 0's) are logicals
        elif _isbinary(obj) and not (obj == 0).all():               return 'logical'
        # General numeric array
        else:                                                       return 'array'
    else:
        raise TypeError("Undetermined type of variable:", obj)

