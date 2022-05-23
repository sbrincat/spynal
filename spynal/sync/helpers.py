# -*- coding: utf-8 -*-
""" Helper functions for `sync` module for oscillatory synchrony analysis """
import numpy as np

from spynal.utils import index_axis
from spynal.spectra.spectra import spectrogram
 

def _infer_data_type(data):
    """ Infer type of data signal given -- 'raw' (real) | 'spectral' (complex) """
    if np.isrealobj(data):  return 'raw'
    else:                   return 'spectral'


def _sync_raw_to_spectral(data1, data2, smp_rate, axis, time_axis, taper_axis,
                          spec_method, data_type, **kwargs):
    """
    Check data input to lfp-lfp synchrony methods.
    Determine if input data is raw or spectral, compute spectral transform if raw.
    """
    if data_type is None: data_type = _infer_data_type(data1)

    assert _infer_data_type(data2) == data_type, \
        ValueError("data1 and data2 must have same data_type (raw or spectral)")

    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, \
            "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, \
            "For raw/time-series data, need to input value for <time_axis>"
        if spec_method == 'multitaper': kwargs.update(keep_tapers=True)
        data1,freqs,timepts = spectrogram(data1, smp_rate, axis=time_axis, method=spec_method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2, smp_rate, axis=time_axis, method=spec_method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        # Account for new frequency (and/or taper) axis prepended before time_axis
        n_new_axes = 2 if spec_method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
        if spec_method == 'multitaper': taper_axis = time_axis-1

    else:
        freqs = []
        timepts = []
        if spec_method == 'multitaper':
            assert taper_axis is not None, \
                ValueError("Must set value for taper_axis for multitaper spectral inputs")

    return data1, data2, freqs, timepts, axis, time_axis, taper_axis


def _sfc_raw_to_spectral(spkdata, lfpdata, smp_rate, axis, time_axis, taper_axis, timepts,
                         method, spec_method, data_type, **kwargs):
    """
    Check data input to spike-field coupling methods.
    Determine if input data is raw or spectral, compute spectral transform if raw.
    """
    if data_type is None:
        spk_data_type = _infer_data_type(spkdata)
        lfp_data_type = _infer_data_type(lfpdata)
        # Spike and field data required to have same type (both raw or spectral) for coherence
        if method == 'coherence':
            assert spk_data_type == lfp_data_type, \
                ValueError("Spiking (%s) and LFP (%s) data must have same data type" % \
                            (spk_data_type,lfp_data_type))
        # Spike data must be raw (not spectral) for phase-based methods
        else:
            assert _infer_data_type(spkdata) == 'raw', \
                ValueError("Spiking data must be given as raw, not spectral, data")

        data_type = lfp_data_type

    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert (timepts is not None) or (smp_rate is not None), \
            ValueError("If no value is input for <timepts>, must input value for <smp_rate>")

        # Ensure spkdata is boolean array with 1's for spiking times (ie not timestamps)
        assert spkdata.dtype != object, \
            TypeError("Spiking data must be converted from timestamps to boolean format")
        spkdata = spkdata.astype(bool)

        # Default timepts to range from 0 - n_timepts/smp_rate
        if timepts is None:     timepts = np.arange(lfpdata.shape[time_axis]) / smp_rate
        elif smp_rate is None:  smp_rate = 1 / np.diff(timepts).mean()

        # For multitaper, keep tapers, to be averaged across like trials below
        if spec_method == 'multitaper': kwargs.update(keep_tapers=True)
        # For multitaper phase sync, spectrogram window spacing must = sampling interval (eg 1 ms)
        if (spec_method == 'multitaper') and (method != 'coherence'):
            kwargs.update(spacing=1/smp_rate)

        # All spike-field coupling methods require spectral data for LFPs
        lfpdata,freqs,times = spectrogram(lfpdata, smp_rate, axis=time_axis, method=spec_method,
                                          data_type='lfp', **kwargs)

        # Coherence requires spectral data for both spikes and LFPs
        if method == 'coherence':
            spkdata,_,_ = spectrogram(spkdata, smp_rate, axis=time_axis, method=spec_method,
                                      data_type='spike', **kwargs)

        timepts_raw = timepts
        timepts = times + timepts[0]

        # Multitaper spectrogram loses window width/2 timepoints at either end of data
        # due to windowing. For phase-based SFC methods, must remove these timepoints
        # from spkdata to match. (Note: not issue for coherence, which operates in spectral domain)
        if (spec_method == 'multitaper') and (method != 'coherence'):
            retained_times = (timepts_raw >= timepts[0]) & (timepts_raw <= timepts[-1])
            spkdata = index_axis(spkdata, time_axis, retained_times)

        # Frequency axis always inserted just before time axis, so if
        # observation/trial axis is later, must increment it
        # Account for new frequency (and/or taper) axis
        n_new_axes = 2 if spec_method == 'multitaper' else 1

        if method != 'coherence':
            # Set up indexing to preserve axes before/after time axis,
            # but insert n_new_axis just before it
            slicer = [slice(None)]*time_axis + \
                    [np.newaxis]*n_new_axes + \
                    [slice(None)]*(spkdata.ndim-time_axis)
            # Insert singleton dimension(s) into spkdata to match freq/taper dim(s) in lfpdata
            spkdata = spkdata[tuple(slicer)]

        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
        if spec_method == 'multitaper': taper_axis = time_axis - 1

    else:
        freqs = []
        # timepts = []

    return spkdata, lfpdata, freqs, timepts, smp_rate, axis, time_axis, taper_axis
