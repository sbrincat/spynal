# neural_analysis
Tools for preprocessing and basic analysis of systems/cognitive neurophysiology data in Python

### Features include:
Covers typical preprocessing and basic analysis steps in neural analysis workflows  
Simple interface: no object-oriented programming or complicated data/params structures  
Consistent Matlab-like function calls: analysis(data, axis, extra_param=default)  
Modular: can use specific functionality w/o requiring an entire processing chain  
Well-documented: function docstrings explain usage and input/output format  
Vectorized mass-univariate analysis: runs in parallel across all channels, no for loops needed  

### Package includes the following modules:
matIO -- Simple, painless loading of MATLAB .mat data files into Python  
spikes -- Basic preprocessing and analyses of neural spiking activity  
spectra -- Spectral analysis and LFP/continuous data preprocessing  
sync -- Analysis of oscillatory neural synchrony (field-field and spike-field)  
info -- Measures of neural information about task/behavior variables  
randstats -- Nonparametric randomization, permutation, and bootstrap stats  
plots -- Generation of common plot types, and plotting utilities  


## Download & installation instructions
### Download repository
    - Open a terminal window, navigate to folder you want repository to live under (cd "parent directory")
    - Run: git clone https://github.com/sbrincat/neural_analysis.git

### Install package
    - Navigate into the newly-created neural_analysis folder (cd "parent directory"/neural_analysis)
    - For end users:  at command line, run: python setup.py install
    - For developers: at command line, run: python setup.py develop
        (this will allow you to edit the library without reinstalling)

You should now be able to directly import neural_analysis modules in your Python code/notebooks like so:  
    import neural_analysis.sync  
    import neural_analysis.spikes as spk  
    from neural_analysis.randstats import two_sample_test  