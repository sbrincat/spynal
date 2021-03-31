# neural_analysis
Tools for basic analysis of systems/cogntive neurophysiology data in Python

Features include:
Simple interface: no object-oriented programming or complicated data/params structures  
No buy-in necessary: can use specific functionality w/o requiring an entire processing chain  
Simple, uniform Matlab-like function call structure: analysis(data, axis, extra_param=default)
Vectorized mass-univariate analysis: input data for all channels at once, no for loop needed

Package includes the following modules:  
spike_analysis -- Basic analyses of neural spiking activity  
neural_synchrony -- Analysis of neural oscillations and synchrony  
neural_info -- Measures of neural information about task/behavior variables  
randomization_stats -- Nonparametric randomization, permutation, and bootstrap stats  
