Quickstart
==========

A quick example of sampling of a workflow for loading data from a Matlab file,
doing time-frequency analysis, and plotting

.. code-block:: python

    from spynal.matIO import loadmat
    from spynal.spectra import spectrogram, plot_spectrogram

    # Load LFP data from a Matlab .mat file
    lfp, timepts = loadmat(datafile.mat, variables=['lfp','timepts'])

    # Compute time-frequency (spectrogram) power using wavelet transform
    spec, freqs, timepts = spectrogram(lfp, smp_rate, axis=0, method='wavelet', spec_type='power')
   
    # Plot time-frequency power as a heatmap
    plot_spectrogram(timepts, freqs, spec, cmap='viridis')

A more extensive tutorial covering all the basic Spynal functionality can be found
in this `notebook <https://github.com/sbrincat/neural_analysis/blob/master/neural_analysis_tutorial.ipynb>`_
