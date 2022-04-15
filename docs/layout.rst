==============
Project layout
==============

This chart shows the layout of all public directories, modules, and submodules in Spynal code

::

    spynal/                         # Top-level folder
    ├── matIO/                          # Loading from/saving to Matlab .mat files
        ├── matIO.py                    # General interface to mat file I/O
        ├── matIO_7.py                      # Matlab I/O for version 7 mat files
        └── matIO_73.py                     # Matlab I/O for version 7.3 mat files
    ├── spikes.py                       # Spiking data preprocessing & analysis            
    ├── spectra/                        # Spectral analysis and LFP preprocessing
        ├── spectra.py                      # General interface to spectral/LFP functions
        ├── wavelet.py                      # Continuous-wavelet analysis
        ├── multitaper.py                   # Multitaper spectral analysis
        └── bandfilter.py                   # Bandpass filter (& Hilbert transform)-based analysis
    ├── sync                            # Oscillatory synchrony analysis (LFP-LFP & spike-LFP)
        ├── sync.py                         # General interface for oscillatory synchrony analysis
        ├── coherence.py                    # Coherence analysis
        ├── phasesync.py                    # Phase-based synchrony analysis
    ├── info.py                         # Measures of neural information
    ├── randstats/                      # Nonparametric randomization statistics
        ├── randstats.py                    # General interface to randomization statistics
        ├── permutation.py                  # Permutation-based statistics
        ├── bootstrap.py                    # Bootstrap-based statistics
    ├── plots.py                        # Plotting functions & utilities
    └── utils.py                        # General purpose utilities
