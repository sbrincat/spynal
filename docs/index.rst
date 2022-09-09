.. Spynal documentation master file, created by
   sphinx-quickstart on Wed Mar 16 16:41:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================================
Spynal -- the Simple Python Neural Analysis Library
===================================================

Spynal is an open-source Python library with tools for loading, preprocessing, analysis,
plotting, and simulation of systems/cognitive neurophysiology data.

Our intended scope covers preprocessing and basic analysis steps in most electrophysiology 
analysis workflows, including analysis of spiking data, LFP/spectral analysis, synchrony,
neural information analysis, randomization stats, and more. We aim to provide a solid 
backbone to facilitate users taking advantage of more advanced analytical tools in Python,
but we also intend to continue adding more complex analyses as well.

Our intended user base is anyone doing analysis of neurophysiology data, but we particularly
aim to be accessible and easy-to-use for people migrating to Python from Matlab or other
similar programming languages.

Design principles
-----------------
The library is based on several underlying design principles:

- **Simplicity** -- Simple, easy-to-use functional inferface that's friendly to users migrating to Python
    from other languages. No knowledge of object-oriented programming or complicated
    data/parameter structures required.
- **Consistency** -- Functions have a uniform, familiar interface of the form:
    ``analysis(data, labels, axis, extra_param=value)``
- **Modularity** -- Users can pick and choose specific functionality without buy-in to an entire ecosystem
    or processing chain. Developers can add new methods as plug-in options.
- **Well-documented** -- Function docstrings explain their usage and the format of
    input arguments and returned outputs
- **Foolproof** -- Function parameters default to values typical for neural analysis.
    Functions have extensive checks for proper inputs.
- **Flexibility** -- Detailed parameterization allows customization of analysis to users' needs
- **Flexible vectorized analysis** -- Most analysis runs in parallel across a specified data axis.
    No need to reshape/transpose data to specific shape or embed analysis in nested *for* loops.

Table of contents
-----------------
.. toctree::
    :maxdepth: 2
    
    quickstart
    installation
    API_reference
    layout
    modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
