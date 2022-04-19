=====================
General API reference
=====================

This section gives a high-level summary of the Spynal API -- how users interface with it,
and how developers should write new code for it. *Specific* reference for each Spynal module
and function can be found in the :doc:`Function reference by module <modules>` section.

Functional interface
--------------------
Spynal has a Matlab-like purely functional interface. No object-oriented programming required,
and no hidden state changes. To perform analysis, users simply call the appropriate function
with a data array, and typically one or more appropriate parameters:

``analysis(data, labels, ...)``

We strive to give our functions and variables succint but intuitive names.

Data types & Parameters
-----------------------
The basic data type used in Spynal is the simple Numpy ndarray (a multi-dimensional array type
similar to Matlab arrays). No specialized variable types, and no complicated parameter structures
are needed. To set options for analysis functions, we employ the standard Python keyword argument
syntax: ``param=value``. Parameters are set to default values typical for the given analysis, but
these can be overriden with custom values to meet users' needs:

``analysis(data, labels, extra_param1=value1, extra_param2=value2, ...)``

Flexible vectorized analysis
----------------------------
Data from different sources often has different dimensionality and dimensional ordering.
Often this means data arrays need to be transposed/reshaped to fit assumptions of an analysis
library, then reshaped back afterward.

Neural data analysis is often performed in "mass-univariate" fashion, with the same analysis
applied to multiple data channels, trials, time points, frequencies, etc. Often this requires
writing a series of nested `for` loops for every analysis, like this::

    for channel in channels:
        for trial in trials:
            results[trial,channel] = analysis(data[trial,channel])

Spynal solves both problems using flexible vectorized analysis. For most functions, users can input
data with *any arbitrary dimensionality*, along with an `axis` parameter telling the function which
array axis to perform computations on (eg time for spectral analysis, trials for information
analysis), and Spynal handles it appropriately under the hood, without further user action. 
Analysis is typically performed in parallel independently (in "mass-univariate" fashion) along all 
other data array axes (eg channels, time points, frequencies, trials, etc.)

``analysis(data, labels, axis, extra_param1=value1, ...)``

In documentation, the required shape for flexibly-shaped data array inputs is described like this:
`shape=(...,n,...)`. The `n` indicates that one array axis must have length `n`, while the 
ellipses `...` indicate the array may have any arbitrary number of other dimensions with
arbitrary lengths.

Hierarchical organization
-------------------------
Spynal functionality is organized into multiple hierarchical levels. At the highest level are
modules, each of which contains all functions for a specific neural signal type (eg `spikes`
for spiking data analysis) or class of analyses (eg `info` for neural information analysis).
Some modules are further divided into submodules.

Many functions themselves also have a hierarchical organization. A top-level function provides
a general interface for a given type of analysis (eg `spectrogram` for time-frequency spectral
analysis, `neural_info` for neural information analysis). These contain parameters used across
all specific methods for this analysis, as well as a `method` parameter, allowing users to
specify which specific method to use (eg 'wavelet' vs 'multitaper' spectrogram, 'pev' vs
'mutual_info' as an information measure).

Each specific method is contained within a lower-level function, which may also contain additional
parameters only used for that specific method. Users are free to access functionality at either of
these levels. And developers only need to match the API of the higher-level method to add new
methods as plug-in options.