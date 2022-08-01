#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Functions for loading from and saving to Matlab v7 (and earlier) MAT files using scipy.io """
import numpy as np

import scipy.io

from spynal.matIO.helpers import _parse_typemap, _is_structured_array, \
                                 _structuredarray_to_dict, _structuredarray_to_dataframe, \
                                 _v7_matlab_type, _process_v7_object, DEBUG


def _load7(filename, variables=None, typemap=None, order='Matlab'):
    """
    Load data variables from a version 7 (or older) Matlab MAT file
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
    """  List data variables from a version 7 (or older) Matlab MAT file """
    # Load list of 3-tuples of (variable,size,type) for each variable in file
    variables = scipy.io.whosmat(filename, appendmat=True, **kwargs)
    # Extract and return just the variable names
    return [vbl[0] for vbl in variables]


def _save7(filename, variables, **kwargs):
    """
    Save data variables to version 7 Matlab MAT file
    Uses scipy.io.savemat to save data
    """
    scipy.io.savemat(filename,variables,**kwargs)

