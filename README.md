# spynal
Simple Python Neural Analysis Library, the backbone of your neural data analysis pipeline.  

Tools for preprocessing and basic analysis of systems/cognitive neurophysiology data in Python  

Covers typical preprocessing and basic analysis steps in neural analysis workflows.
Intended users are anyone doing analysis of neurophysiology data, but we particularly aim to be
accessible and easy-to-use for people migrating to Python from Matlab or other similar
programming languages.  

### Features include:
**Simplicity**: easy-to-use interface; no OOP or complicated data/params structures  
**Consistency**: consistent function signature ~ analysis(data, axis, extra_param=default)  
**Modularity**: can use specific functionality w/o requiring an entire processing chain  
**Foolproof**: extensive documentation and checks for proper inputs  
**Flexibility**: detailed parameterization allows customization of analysis to users' needs  
**Vectorized mass-univariate analysis**: runs in parallel across all channels, no loops needed  

### Package includes the following modules:
**matIO** -- Painless loading of MATLAB .mat data files into appropriate Python data structures  
**spikes** -- Preprocessing and analyses of spiking data  
**spectra** -- Spectral analysis and preprocessing of continuous data (LFP/EEG)   
**sync** -- Analysis of oscillatory neural synchrony (field-field and spike-field)  
**info** -- Measures of neural information about task/behavior variables  
**randstats** -- Nonparametric randomization, permutation, and bootstrap statistics  
**plots** -- Generation of common plot types; plotting utilities  
**utils** -- Numerical and general purpose utilities  


## Download & installation instructions

### pip install
    - Open a terminal window, navigate to folder you want repository to live under (cd "parent directory")
    - Run: pip install spynal

### Install from source (Github)

#### Download repository
    - Open a terminal window, navigate to folder you want repository to live under (cd "parent directory")
    - Run: git clone https://github.com/sbrincat/spynal.git

#### Install package
    - Navigate into the newly-created spynal folder (cd "parent directory"/spynal)
    - For end users:  at command line, run: python setup.py install
    - For developers: at command line, run: python setup.py develop
        (this will allow you to edit the library without reinstalling)

## Usage

You should now be able to directly import spynal and its modules/functions in your code/notebooks like so:  
    import spynal  
    import spynal as spy  
    import spynal.spectra as spectra  
    from spynal.spectra import spectrogram  