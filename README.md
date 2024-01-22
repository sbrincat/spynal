[![DOI](https://zenodo.org/badge/296102080.svg)](https://zenodo.org/badge/latestdoi/296102080)

# spynal: Simple Python Neural Analysis Library
Tools for preprocessing and basic analysis of systems/cognitive neurophysiology data in Python  

Covers typical preprocessing and basic analysis steps in neural analysis workflows.
Intended users are anyone doing analysis of neurophysiology data, but we particularly aim to be
accessible and easy-to-use for people migrating to Python from Matlab or other similar
programming languages.  

### Features include:
**Simplicity**: Easy-to-use procedural interface; no OOP or complicated data/params structures  
**Consistency**: Consistent function signature ~ `analysis(data, labels, axis, extra_param=value)`  
**Modularity**: Can use specific functionality without buy-in to an entire processing chain  
**Foolproof**: Extensive documentation and checking for proper function inputs  
**Flexibility**: Detailed parameterization allows customization of analysis to users' needs  
**Vectorized analysis**: Runs in parallel across all channels, no loops or data reshaping needed  

### Package includes the following modules:
**matIO** — Painless loading of MATLAB .mat data files into appropriate Python data structures  
**spikes** — Preprocessing and analyses of spiking data  
**spectra** — Spectral analysis and preprocessing of continuous data (LFP/EEG)   
**sync** — Analysis of oscillatory neural synchrony (field-field and spike-field)  
**info** — Measures of neural information about task/behavior variables  
**randstats** — Nonparametric randomization, permutation, and bootstrap statistics  
**plots** — Generation of common plot types; plotting utilities  
**utils** — Numerical and general purpose utilities  

Full documentation can be found at: https://spynal.readthedocs.io/en/latest/  


## Download & installation instructions
Users can either install the most recent stable release using pip or
install the latest updates from the Github repository

### Option 1: pip install
Recommended for users who prefer a simple installation and don't require the latest updates

    - Open a terminal window, navigate to folder you want repository to live under (cd "parent directory")
    - Run: pip install spynal

### Option 2: Install from source (Github)
Recommended for developers and for users that would like access to the latest updates

#### 2a: Download repository
    - Open a terminal window, navigate to folder you want repository to live under (cd "parent directory")
    - Run: git clone https://github.com/sbrincat/spynal.git

#### 2b: Install package
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

## Contributions

We welcome any contributions of new/expanded functionality or more efficient implementations,  
and all issues/bug reports.  
Submit a pull request/issue on Github or email me at [Github handle] AT [institution] .edu  