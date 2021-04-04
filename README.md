# neural_analysis
Tools for basic analysis of systems/cognitive neurophysiology data in Python

Features include:  
Simple interface: no object-oriented programming or complicated data/params structures  
No buy-in necessary: can use specific functionality w/o requiring an entire processing chain  
Simple, uniform Matlab-like function call structure: analysis(data, axis, extra_param=default)  
Vectorized mass-univariate analysis: analysis runs in parallel across all channels, no for loops needed  

Package includes the following modules:  
spikes -- Basic preprocessing and analyses of neural spiking activity  
spectra -- Spectral analysis and LFP/continuous data preprocessing 
sync -- Analysis of oscillatory neural synchrony
info -- Measures of neural information about task/behavior variables  
randstats -- Nonparametric randomization, permutation, and bootstrap stats 
