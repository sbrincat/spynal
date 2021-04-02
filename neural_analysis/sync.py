# -*- coding: utf-8 -*-
"""
sync    A module for analysis of neural synchrony

FUNCTIONS
### Field-field synchrony ###
synchrony           Synchrony between pair of channels using given method

coherence           Time-frequency coherence between pair of channels
ztransform_coherence Z-transform coherence so ~ normally distributed

phase_locking_value Phase locking value (PLV) between pair of channels
pairwise_phase_consistency Pairwise phase consistency (PPC) btwn pair of channels

### Data simulation ###
simulate_oscillation Generates simulated oscillation-in-noise data


DEPENDENCIES
pyfftw              Python wrapper around FFTW, the speedy FFT library
spikes              A module for basic analyses of neural spiking activity

Created on Thu Oct  4 15:28:15 2018

@author: sbrincat
"""

import os
import time
from warnings import warn
from math import floor,ceil,log2,pi,sqrt
from collections import OrderedDict
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal.windows import dpss
from scipy.signal import filtfilt,hilbert,zpk2tf,butter,ellip,cheby1,cheby2
from scipy.stats import norm,mode
 
from .spectra import spectrogram, simulate_oscillation

# =============================================================================
# Field-Field Synchrony functions
# =============================================================================
def synchrony(data1, data2, axis=0, method='PPC', spec_method='wavelet', **kwargs):
    """
    Computes pairwise synchrony between pair of channels of continuous raw or
    spectral (time-frequency) data, using given estimation method

    sync,freqs,timepts= synchrony(data1,data2,axis=0,method='PPC',
                                  spec_method='wavelet',**kwargs)
                                  
    Convenience wrapper function around specific synchrony estimation functions
    coherence, phase_locking_value, pairwise_phase_consistency

    ARGS
    data1,2 (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have arbitrary shape, with
            analysis performed in mass-bivariate fashion independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Scalar. Axis corresponding to distinct observations/trials. Default: 0

    method  String. Synchrony estimation method. Options: 'PPC' [default] | 'PLV' | 'coherence'
            
    **kwargs    All other kwargs passed as-is to synchrony estimation function;=.
            See there for details.

    RETURNS
    sync    ndarray. Synchrony between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in coh (only for raw data)
    timepts (n_timepts,). List of timepoints in coh (only for raw data)

    """
    if method.lower() in ['ppc','pairwise_phase_consistency']:  sync_fun = pairwise_phase_consistency
    elif method.lower() in ['plv','phase_locking_value']:       sync_fun = phase_locking_value
    elif method.lower() in ['coh','coherence']:                 sync_fun = coherence
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)
    
    return sync_fun(data1, data2, axis=axis, method=spec_method, **kwargs)
    
        
def coherence(data1, data2, axis=0, return_phase=False, single_trial=None, ztransform=False,
              method='wavelet', data_type=None, smp_rate=None, time_axis=None, taper_axis=None,
              **kwargs):
    """
    Computes pairwise coherence between pair of channels of raw or
    spectral (time-frequency) data (LFP or spikes)

    coh,freqs,timepts[,dphi] = coherence(data1,data2,axis=0,return_phase=False,
                                         single_trial=None,ztransform=False,
                                         method='wavelet',data_type=None,smp_rate=None,
                                         time_axis=None,taper_axis=None,**kwargs)

    ARGS
    data1,2 (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have arbitrary shape, with
            analysis performed in mass-bivariate fashion independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Int. Axis corresponding to distinct observations/trials. Default: 0

    return_phase Bool. If True, returns additional output with mean phase difference

    single_trial String or None. What type of coherence estimator to compute:
                None        standard across-trial estimator [default]
                'pseudo'    single-trial estimates using jackknife pseudovalues
                'richter'   single-trial estimates using actual jackknife estimates
                            as in Richter & Fries 2015

    ztransform  Bool. If True, returns z-transformed coherence using Jarvis &
                Mitra (2001) method. If false [default], returns raw coherence.

    data_type   Str. What kind of data are we given in data1,data2:
                'raw' or 'spectral'
                Default: assume 'raw' if data is real; 'spectral' if complex

    Following args are only used for spectral analysis for data_type == 'raw'

    method      String. Spectral method. 'wavelet' [default] | 'multitaper'

    smp_rate    Scalar. Sampling rate of data (only needed for raw data)

    time_axis   Int. Axis of data corresponding to time (ONLY needed for raw data)

    taper_axis  Int. Axis of spectral data corresponding to tapers (ONLY needed for 
                multitaper spectral data)

    Any other kwargs passed as-is to spectrogram() function.

    RETURNS
    coh     ndarray. Magnitude of coherence between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in coh (only for raw data)
    timepts (n_timepts,). List of timepoints in coh (only for raw data)

    dphi   ndarray. Mean phase difference between data1 and data2 in radians.
           Positive values correspond to data1 leading data2.
           Negative values correspond to data1 lagging behind data2.
           Optional: Only returned if return_phase is True.

    REFERENCE
    Single-trial method:    Womelsdorf, Fries, Mitra, Desimone (2006) Science
    Single-trial method:    Richter, ..., Fries (2015) NeuroImage
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")
            
    assert (single_trial is None) or (single_trial in ['pseudo','richter']), \
        ValueError("Parameter <single_trial> must be = None, 'pseudo', or 'richter'")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    if data_type is None: data_type = _infer_data_type(data1)
    
    # If raw data is input, compute spectral transform first
    # print(axis, time_axis, data1.shape, data1.mean(), data2.mean())
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        if method == 'multitaper': kwargs.update(keep_tapers=True)
        data1,freqs,timepts = spectrogram(data1, smp_rate, axis=time_axis, method=method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2, smp_rate, axis=time_axis, method=method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        # Account for new frequency (and/or taper) axis prepended before time_axis
        n_new_axes = 2 if method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
        if method == 'multitaper': taper_axis = time_axis-1
        
    else:
        freqs = []
        timepts = []
        if method == 'multitaper':
            assert taper_axis is not None, \
                ValueError("Must set value for taper_axis for multitaper spectral inputs")

    # For multitaper, compute means across trials, tapers; df = 2*n_trials*n_tapers
    if method == 'multitaper':
        reduce_axes = (axis,taper_axis)
        df = 2*data1.shape[axis]*data1.shape[taper_axis]
    # Otherwise, just compute means across trials; df = 2*n_trials (TODO is this true?)
    else:
        reduce_axes = axis
        df = 2*data1.shape[axis]


    def _spec_to_coh(data1, data2, axis, return_phase):
        """ Compute coherence from a pair of spectra/spectrograms """
        # Compute auto-spectra, average across trials/tapers
        # Note: .real deals with floating point error, converts complex dtypes to float        
        S1  = np.mean(data1*data1.conj(), axis=axis).real
        S2  = np.mean(data2*data2.conj(), axis=axis).real
        
        if return_phase:
            # Compute cross spectrum, average across trials/tapers
            S12 = np.mean(data1*data2.conj(), axis=axis)
            
            # Calculate complex coherency as cross-spectrum / product of spectra
            coherency = S12 / np.sqrt(S1*S2)

            # Absolute value converts complex coherency -> coherence
            # Angle extracts mean coherence phase angle
            # Note: .real deals with floating point error, converts complex dtypes to float
            return np.abs(coherency).real, np.angle(coherency) 
        
        else:
            # Compute cross spectrum, average across trials/tapers,
            # and take absolute value (bc phase not needed)
            S12 = np.abs(np.mean(data1*data2.conj(), axis=axis)).real

            # Calculate coherence as cross-spectrum / product of spectra
            return S12 / np.sqrt(S1*S2)
    
    
    def _csd_to_coh(S12, S1, S2, axis):
        """ Compute coherence from cross spectrum, auto-spectra """
        # Average cross and individual spectra across observations/trials
        S12 = np.abs(np.mean(S12, axis=axis)).real
        S1  = np.mean(S1, axis=axis).real
        S2  = np.mean(S2, axis=axis).real
        # Calculate coherence as cross-spectrum / product of spectra
        return S12 / np.sqrt(S1*S2)
        
    # Standard across-trial coherence estimator
    if single_trial is None:
        if return_phase:    coh, dphi = _spec_to_coh(data1, data2, reduce_axes, return_phase)
        else:               coh = _spec_to_coh(data1, data2, reduce_axes, return_phase)
        
        if ztransform: coh = ztransform_coherence(coh,df)

    # Single-trial coherence estimator using jackknife resampling method
    else:
        # If observation axis != 0, permute axis to make it so
        if axis != 0:
            data1 = np.moveaxis(data1,axis,0)
            data2 = np.moveaxis(data2,axis,0)
        n = data1.shape[0]

        # Compute cross spectrum and auto-spectra for each observation/trial
        S12 = data1*data2.conj()
        S1  = data1*data1.conj()
        S2  = data2*data2.conj()

        coh = np.zeros_like(data1,dtype=float)

        # Do jackknife resampling -- estimate statistic w/ each observation left out
        # (this is the 'richter' estimator)
        trials = np.arange(n)
        for trial in trials:
            idxs = trials != trial
            coh[trial,...] = _csd_to_coh(S12[idxs,...],S1[idxs,...],S2[idxs,...],0)

        if ztransform: coh = ztransform_coherence(coh,df/n)

        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            coh_full = _csd_to_coh(S12,S1,S2,0)
            if ztransform: coh_full = ztransform_coherence(coh_full,df)
            coh = jackknife_to_pseudoval(coh_full[np.newaxis,...],coh,n)

        # If observation axis wasn't 0, permute axis back to original position
        if axis != 0: coh = np.moveaxis(coh,0,axis)

    if return_phase:    return coh, freqs, timepts, dphi
    else:               return coh, freqs, timepts


def ztransform_coherence(coh, df, beta=23/20):
    """
    z-transforms coherence values to render them approximately normally distributed

    z = ztransform_coherence(coh,df,beta=23/20)

    ARGS
    coh     ndarray. Raw coherence values.

    df      Int. Degrees of freedom of coherence estimates. For multitaper
            estimates, this is usually df = 2*n_trials*n_tapers; for other
            estimates, this is usually df = 2*n_trials

    beta    Scalar. Mysterious number from Jarvis & Mitra to make output z-scores
            approximately normal. Default: 23/20 (see J & M)

    RETURNS
    z       ndarray of same shape as <coh>. z-transformed coherence

    REFERENCE
    Jarvis & Mitra (2001) Neural Computation
    Hipp, Engel, Siegel (2011) Neuron
    """
    return beta*(np.sqrt(-(df-2)*np.log(1-coh**2)) - beta)


def phase_locking_value(data1, data2, axis=0, return_phase=False,
                        single_trial=None, method='wavelet', data_type=None,
                        smp_rate=None, time_axis=None, taper_axis=None, **kwargs):
    """
    Computes phase locking value (PLV) between raw or spectral (time-frequency) LFP data

    PLV is the mean resultant length (magnitude of the vector mean) of phase
    differences dphi btwn phases of data1 and data2:
        dphi = phase(data1) - phase(data2)
        plv  = abs( trialMean(exp(i*dphi)) )

    plv,freqs,timepts[,dphi] = phase_locking_value(data1,data2,axis=0,return_phase=False,
                                                 single_trial=None,
                                                 method='wavelet',data_type=None,
                                                 smp_rate=None,time_axis=None,
                                                 taper_axis=None,**kwargs)

    ARGS
    data1,2 (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have
            Can have arbitrary shape, with analysis performed independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Int. Axis corresponding to distinct observations/trials. Default: 0

    return_phase Bool. If True, returns additional output with mean phase difference

    single_trial String or None. What type of coherence estimator to compute:
            None        standard across-trial estimator [default]
            'pseudo'    single-trial estimates using jackknife pseudovalues
            'richter'   single-trial estimates using actual jackknife estimates
                        as in Richter & Fries 2015

    Following args are only used for spectral analysis for data_type == 'raw'

    method      String. Spectral method. 'wavelet' [default] | 'multitaper'

    data_type   Str. What kind of data are we given in data1,data2: 'raw' or 'spectral'
                Default: assume 'raw' if data is real; 'spectral' if complex

    smp_rate    Scalar. Sampling rate of data (only needed for raw data)

    time_axis   Int. Axis of data corresponding to time (only needed for raw data)

    taper_axis  Int. Axis of spectral data corresponding to tapers (ONLY needed for 
                multitaper spectral data)

    Any other kwargs passed as-is to spectrogram() function.

    RETURNS
    plv     ndarray. Phase locking value between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in plv (only for raw data)
    timepts (n_timepts,). List of timepoints in plv (only for raw data)

    dphi   ndarray. Mean phase difference between data1 and data2 in radians.
           Positive values correspond to data1 leading data2.
           Negative values correspond to data1 lagging behind data2.
           Optional: Only returned if return_phase is True.

    REFERENCES
    Lachaux et al. (1999) Human Brain Mapping
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    n_obs    = data1.shape[axis]

    if data_type is None: data_type = _infer_data_type(data1)
    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        if method == 'multitaper': kwargs.update(keep_tapers=True)
        
        data1,freqs,timepts = spectrogram(data1,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp', spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp', spec_type='complex', **kwargs)
        
        # Account for new frequency (and/or taper) axis
        n_new_axes = 2 if method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
        if method == 'multitaper': taper_axis = time_axis-1
        
    else:
        freqs = []
        timepts = []
        if method == 'multitaper':
            assert taper_axis is not None, \
                ValueError("Must set value for taper_axis for multitaper spectral inputs")

    # For multitaper, compute means across trials, tapers
    if method == 'multitaper':  reduce_axes = (axis,taper_axis)
    # Otherwise, just compute means across trials
    else:                       reduce_axes = axis
        
        
    def _spec_to_plv(data1, data2, axis, return_phase, keepdims):
        """ Compute PLV from a pair of spectra/spectrograms """
        # Cross-spectrum-based method adapted from FieldTrip ft_conectivity_ppc()
        # Note: circular mean-based algorithm is ~3x slower
        csd = data1*data2.conj()    # Compute cross-spectrum
        csd = csd / np.abs(csd)     # Normalize cross-spectrum
        if return_phase:
            # Compute vector mean across trial/observations
            vector_mean = np.mean(csd,axis=axis,keepdims=keepdims)
            # Compute PLV, phase difference as absolute value, angle of vector mean
            return np.abs(vector_mean), np.angle(vector_mean)        
        else:
            # Compute vector mean across trial/observations -> absolute value
            return np.abs(np.mean(csd,axis=axis,keepdims=keepdims))
        
                
    # Standard across-trial PLV estimator
    if single_trial is None:
        if return_phase:
            plv,dphi = _spec_to_plv(data1,data2,reduce_axes,return_phase,False)
            return  plv, freqs, timepts, dphi

        else:
            plv = _spec_to_plv(data1,data2,reduce_axes,return_phase,False)
            return  plv, freqs, timepts

    # Single-trial PLV estimator using jackknife resampling method
    else:
        # Note: two_sample_jackknife() (temporarily) shifts trial axis to 0, so axis=0 here
        jackfunc = lambda data1,data2: _spec_to_plv(data1,data2,0,False,False)
        # Jackknife resampling of PLV statistic (this is the 'richter' estimator)
        plv = two_sample_jackknife(jackfunc,data1,data2,axis=reduce_axes)
        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            plv_full = _spec_to_plv(data1,data2,reduce_axes,False,True)
            plv = jackknife_to_pseudoval(plv_full,plv,n_obs)

        return  plv, freqs, timepts


def pairwise_phase_consistency(data1, data2, axis=0, return_phase=False,
                               single_trial=None, method='wavelet',
                               data_type=None, smp_rate=None, time_axis=None,
                               taper_axis=None, **kwargs):
    """
    Computes pairwise phase consistency (PPC) between raw or spectral
    (time-frequency) LFP data, which is unbiased by n (unlike PLV and coherence)

    PPC is an unbiased estimator of PLV^2, and can be expressed (and computed
    efficiently) in terms of PLV and n:
        PPC = (n*PLV^2 - 1) / (n-1)

    ppc,freqs,timepts[,dphi] = pairwise_phase_consistency(data1,data2,axis=0,
                                                          return_phase=False,single_trial=None,
                                                          method='wavelet',data_type=None,
                                                          smp_rate=None,time_axis=None,
                                                          taper_axis=None,**kwargs)

    ARGS
    data1,data2   (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have
            Can have arbitrary shape, with analysis performed independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis        Int. Axis corresponding to distinct observations/trials. Default: 0

    return_phase Bool. If True, returns additional output with mean phase difference

    single_trial String or None. What type of estimator to compute:
                None        standard across-trial estimator [default]
                'pseudo'    single-trial estimates using jackknife pseudovalues
                'richter'   single-trial estimates using actual jackknife estimates
                            as in Richter & Fries 2015

    Following args are only used for spectral analysis for data_type == 'raw'

    method      String. Spectral method. 'wavelet' [default] | 'multitaper'

    data_type   Str. What kind of data are we given in data1,data2: 'raw' or 'spectral'
                Default: assume 'raw' if data is real; 'spectral' if complex

    smp_rate    Scalar. Sampling rate of data (only needed for raw data)

    time_axis   Int. Axis of raw data corresponding to time (ONLY needed for raw data)

    taper_axis  Int. Axis of spectral data corresponding to tapers (ONLY needed for 
                multitaper spectral data)
                
    **kwargs    Any other keyword args passed as-is to spectrogram() function.

    RETURNS
    ppc     Pairwise phase consistency between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in ppc (only for raw data)
    timepts (n_timepts,). List of timepoints in ppc (only for raw data)

    dphi    ndarray. Mean phase difference between data1 and data2 in radians.
            Positive values correspond to data1 leading data2.
            Negative values correspond to data1 lagging behind data2.
            Optional: Only returned if return_phase is True.

    REFERENCES
    Original concept:   Vinck et al. (2010) NeuroImage
    Relation to PLV:    Kornblith, Buschman, Miller (2015) Cerebral Cortex
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")
    assert (single_trial is None) or (single_trial in ['pseudo','richter']), \
        ValueError("Parameter <single_trial> must be = None, 'pseudo', or 'richter'")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    n_obs    = data1.shape[axis]

    if data_type is None: data_type = _infer_data_type(data1)
    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        if method == 'multitaper': kwargs.update(keep_tapers=True)
        
        data1,freqs,timepts = spectrogram(data1,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp',spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp',spec_type='complex', **kwargs)
        
        # Account for new frequency (and/or taper) axis
        n_new_axes = 2 if method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
        if method == 'multitaper': taper_axis = time_axis-1
        
    else:
        freqs = []
        timepts = []
        if method == 'multitaper':
            assert taper_axis is not None, \
                ValueError("Must set value for taper_axis for multitaper spectral inputs")        

    # For multitaper, compute means across trials, tapers
    if method == 'multitaper':  reduce_axes = (axis,taper_axis)
    # Otherwise, just compute means across trials
    else:                       reduce_axes = axis
    
    
    def _spec_to_ppc(data1, data2, axis, return_phase, keepdims):
        """ Compute PPC from a pair of spectra/spectrograms """
        if np.isscalar(axis):   n = data1.shape[axis]
        else:                   n = np.prod([data1.shape[ax] for ax in axis])        
        
        # Compute PLV
        csd = data1*data2.conj()    # Compute cross-spectrum
        csd = csd / np.abs(csd)     # Normalize cross-spectrum
                        
        if return_phase:
            # Compute vector mean across trial/observations
            vector_mean = np.mean(csd,axis=axis,keepdims=keepdims)
            # Compute PLV, phase difference as absolute value, angle of vector mean
            plv, dphi = np.abs(vector_mean), np.angle(vector_mean)
            return plv_to_ppc(plv, n), dphi
        else:
            # Compute vector mean across trial/observations -> absolute value
            plv = np.abs(np.mean(csd,axis=axis,keepdims=keepdims))
            return plv_to_ppc(plv, n)
        
        
    # Standard across-trial PPC estimator
    if single_trial is None:
        if return_phase:
            ppc,dphi = _spec_to_ppc(data1,data2,reduce_axes,return_phase,False)
            return ppc, freqs, timepts, dphi

        else:
            ppc = _spec_to_ppc(data1,data2,reduce_axes,return_phase,False)
            return ppc, freqs, timepts

    # Single-trial PPC estimator using jackknife resampling method
    else:
        # Note: two_sample_jackknife() (temporarily) shifts trial axis to 0, so axis=0 here
        jackfunc = lambda data1,data2: _spec_to_ppc(data1,data2,0,False,False)
        # Jackknife resampling of PPC statistic (this is the 'richter' estimator)
        ppc = two_sample_jackknife(jackfunc,data1,data2,axis=reduce_axes)
        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            ppc_full = _spec_to_ppc(data1,data2,reduce_axes,False,True)
            ppc = jackknife_to_pseudoval(ppc_full,ppc,n_obs)

        return ppc, freqs, timepts


def plv_to_ppc(plv, n):
    """ Converts PLV to PPC as PPC = (n*PLV^2 - 1)/(n-1) """
    return (n*plv**2 - 1) / (n - 1)


# =============================================================================
# Utility functions for generating single-trial jackknife pseudovalues
# =============================================================================
def two_sample_jackknife(func, data1, data2, *args, axis=0, **kwargs):
    """
    Jackknife resampling of arbitrary statistic
    computed by <func> on <data1> and <data2>

    ARGS
    func    Callable. Takes data1,data2 (and optionally any *args and **kwargs) as input,
            returns statistic to jackknife

    data1,2 ndarrays of same arbitrary shape. Data to compute jackknife resamples of

    axis    Int. Observation (trial) axis of data1 and data2. Default: 0

    *args,**kwargs  Any additional arguments passed directly to <func>

    RETURNS
    stat    ndarray of same size as data1,data2. Jackknife resamples of statistic.
    """

    # If observation axis != 0, permute axis to make it so
    if axis != 0:
        data1 = np.moveaxis(data1,axis,0)
        data2 = np.moveaxis(data2,axis,0)

    n = data1.shape[0]

    stat = np.zeros_like(data1)

    # Do jackknife resampling -- estimate statistic w/ each observation left out
    trials = np.arange(n)
    for trial in trials:
        idxs = trials != trial
        stat[trial,...] = func(data1[idxs,...],data2[idxs,...],*args,**kwargs)

    # If observation axis wasn't 0, permute axis back to original position
    if axis != 0: stat = np.moveaxis(stat,0,axis)

    return stat


def jackknife_to_pseudoval(x, xjack, n):
    """
    Calculate single-trial pseudovalues from leave-one-out jackknife estimates

    pseudo = jackknife_to_pseudoval(x,xjack,axis=0)

    ARGS
    x       ndarray of arbitrary shape (observations/trials on <axis>).
            Statistic (eg coherence) computed on full observed data

    xjack   ndarray of arbitrary shape (must be same shape as x).
            Statistic (eg coherence) computed on jackknife resampled data

    n       Int. Number of observations (trials) used to compute x

    RETURNS
    pseudo  ndarray of same shape as x. Single-trial jackknifed pseudovalues.
    """
    return n*x - (n-1)*xjack


# =============================================================================
# Data simulation and testing functions
# =============================================================================
def simulate_multichannel_oscillation(n_chnls, *args, **kwargs):
    """
    Generates synthetic multichannel data with oscillations at given parameters.
    
    For each channel, generates multiple trials with constant oscillatory signal +  
    random additive Gaussian noise. Parameters can be shared across channels or set
    independently for each channel
    
    data = simulate_multichannel_oscillation(n_chnls, *args, **kwargs)
    
    ARGS
    n_chnls     Int. Number of channels to simulate
    
    *args       Rest of place and keyword arguments are passed to simulate_oscillation()
    **kwargs    for each channel. Each argument can be given in one of two forms:
                (1) A single value equivalent to the same argument to simulate_oscillation().
                    That same value will be used for all n_chnls channels.  
                (2) (n_chnls,) list can be given with different values for each channel,
                    which will be iterated through.
                See simulate_oscillation() for details on arguments.
                
                Exceptions: If a value is set for <seed>, it will be used to set the 
                random number generator only ONCE at the start of this function, so that
                the generation of all channel signals follow a reproducible random sequence.
                
                Simulated data must have same shape for all channels, so all channels must 
                have the same value set for time_range, smp_rate, and n_trials.                
    
    RETURNS
    data        (n_timepts,n_trials,n_chnls) ndarray. Simulated multichannel data.
    """
    # Set a single random seed, so simulation of all multichannel data follows a reproducible sequence
    seed = kwargs.pop('seed',None)    
    if seed is not None:
        if not np.isscalar(seed) and (len(seed) > 1):
            seed = seed[0]
            print("Using only first value given for <seed> (%d). Multiple values not permitted" % seed)
        np.random.seed(seed)
    
    # Ensure all channels have same values for these parameters that determine data size
    for param in ['time_range','smp_rate','n_trials']:
        assert (param not in kwargs) or np.isscalar(kwargs[param]) or np.allclose(np.diff(kwargs[param]), 0), \
            ValueError("All simulated channels must have same value for '%s'" % param)
                
    # Replicate any single-valued arguments (args and kwargs) into (n_chnls,) lists
    args = list(args)
    for i,value in enumerate(args):
        if np.isscalar(value) or (len(value) != n_chnls): args[i] = n_chnls * [value]
        
    for key,value in kwargs.items():
        if np.isscalar(value) or (len(value) != n_chnls): kwargs[key] = n_chnls * [value]
        
    # Simulate oscillatory data for each channel        
    for chnl in range(n_chnls):
        # Extract the i-th value for each arg and kwarg
        chnl_args = [value[chnl] for value in args]
        chnl_kwargs = {key:value[chnl] for key,value in kwargs.items()}

        chnl_data = simulate_oscillation(*chnl_args, **chnl_kwargs)
        
        # HACK Create the array to hold all channel data after we know the shape
        if chnl == 0:
            n_timepts,n_trials = chnl_data.shape
            data = np.empty((n_timepts,n_trials,n_chnls))
            
        data[:,:,chnl] = chnl_data
        
    return data

            
# =============================================================================
# Data reshaping helper functions
# =============================================================================
def _index_axis(data, axis, idxs):
    """ 
    Utility to dynamically index into a arbitrary axis of an ndarray 
    
    data = _index_axis(data, axis, idxs)
    
    ARGS
    data    ndarray. Array of arbitrary shape, to index into given axis of.
    
    axis    Int. Axis of ndarray to index into.
    
    idxs    (n_selected,) array-like of int | (axis_len,) array-like of bool | slice object
            Indexing into given axis of array, given either as list of
            integer indexes or as boolean vector.
    
    RETURNS
    data    ndarray. Input array with indexed values selected from given axis.
    """
    # Generate list of slices, with ':' for all axes except <idxs> for <axis>
    slices = _axis_slices(axis, idxs, data.ndim)

    # Use slices to index into data, and return sliced data
    return data[slices]


def _axis_slices(axis, idxs, ndim):
    """
    Generate list of slices, with ':' for all axes except <idxs> for <axis>,
    to use for dynamic indexing into an arbitary axis of an ndarray
    
    slices = _axis_slices(axis, idxs, ndim)
    
    ARGS
    axis    Int. Axis of ndarray to index into.
    
    idxs    (n_selected,) array-like of int | (axis_len,) array-like of bool | slice object
            Indexing into given axis of array, given either as list of
            integer indexes or as boolean vector.
    
    ndim    Int. Number of dimensions in ndarray to index into
    
    RETURNS
    slices  Tuple of slices. Index tuple to use to index into given 
            axis of ndarray as: selected_values = array[slices]  
    """
    # Initialize list of null slices, equivalent to [:,:,:,...]
    slices = [slice(None)] * ndim

    # Set slice for <axis> to desired indexing
    slices[axis] = idxs

    # Convert to tuple bc indexing arrays w/ a list is deprecated
    return tuple(slices)


def _reshape_data(data, axis=0):
    """
    Reshapes multi-dimensional data array to 2D (matrix) form for analysis

    data, data_shape = _reshape_data(data,axis=0)

    ARGS
    data    (...,n,...) ndarray. Data array of arbitrary shape.

    axis    Int. Axis of data to move to axis 0 for subsequent analysis. Default: 0

    RETURNS
    data    (n,m) ndarray. Data array w/ <axis> moved to axis=0, 
            and all axes > 0 unwrapped into single dimension, where 
            m = prod(shape[1:])

    data_shape (data.ndim,) tuple. Original shape of data array
    """
    if axis < 0: axis = data.ndim + axis    
    
    # Save original shape/dimensionality of <data>
    data_ndim  = data.ndim
    data_shape = data.shape

    if ~data.flags.c_contiguous:
        # If observation axis != 0, permute axis to make it so
        if axis != 0:       data = np.moveaxis(data,axis,0)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix
        if data_ndim > 2:   data = np.reshape(data,(data_shape[axis],-1),order='F')

    # Faster method for c-contiguous arrays
    else:
        # If observation axis != last dim, permute axis to make it so
        if axis != data_ndim - 1: data = np.moveaxis(data,axis,-1)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix, then transpose
        if data_ndim > 2:   data = np.reshape(data,(-1,data_shape[axis]),order='C').T
        else:               data = data.T

    return data, data_shape


def _unreshape_data(data, data_shape, axis=0):
    """
    Reshapes data array from unwrapped 2D (matrix) form back to ~ original
    multi-dimensional form

    data = _unreshape_data(data,data_shape,axis=0)

    ARGS
    data    (axis_len,m) ndarray. Data array w/ <axis> moved to axis=0, 
            and all axes > 0 unwrapped into single dimension, where 
            m = prod(shape[1:])

    data_shape (data.ndim,) tuple. Original shape of data array

    axis    Int. Axis of original data moved to axis 0, which will be shifted 
            back to original axis.. Default: 0

    RETURNS
    data    (...,axis_len,...) ndarray. Data array reshaped back to original shape
    """
    data_shape  = np.asarray(data_shape)
    if axis < 0: axis = data.ndim + axis    

    data_ndim   = len(data_shape) # Number of dimensions in original data
    axis_len    = data.shape[0]   # Length of dim 0 (will become dim <axis> again)

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    # (but with length of dimension <axis> = <axis_length>)
    if data_ndim > 2:
        # Reshape data -> (axis_len,<original shape w/o <axis>>)
        shape = (axis_len,*data_shape[np.arange(data_ndim) != axis])
        # Note: I think you want the order to be 'F' regardless of memory layout
        # TODO test this!!!
        data  = np.reshape(data,shape,order='F')

    # If <axis> wasn't 0, move axis back to original position
    if axis != 0: data = np.moveaxis(data,0,axis)

    return data


def _unreshape_data_newaxis(data,data_shape,axis=0):
    """
    Reshapes data array from unwrapped form back to ~ original
    multi-dimensional form in special case where a new frequency axis was
    inserted before time axis (<axis>)

    data = _unreshape_data_newaxis(data,data_shape,axis=0)

    ARGS
    data    (axis_len,m) ndarray. Data array w/ all axes > 0 unwrapped into
            single dimension, where m = prod(shape[1:])

    data_shape Tuple. Original shape of data array

    axis    Int. Axis of original data corresponding to distinct observations,
            which has become axis 1, but will be permuted back to original axis.
            Default: 0

    RETURNS
    data       (...,n_freqs,n_timepts,...) ndarray. Data array reshaped back to original shape
    """
    data_shape  = np.asarray(data_shape)
    if axis < 0: axis = len(data_shape) + axis    

    data_ndim   = len(data_shape) # Number of dimensions in original data
    n_freqs      = data.shape[0]
    n_timepts    = data.shape[1]

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    if len(data_shape) > 2:
        shape   = (n_freqs, n_timepts, *data_shape[np.arange(data_ndim) != axis])
        data    = np.reshape(data,shape,order='F')

    # If <axis> wasn't 0, move axis back to original position
    if axis != 0: data = np.moveaxis(data,(0,1),(axis,axis+1))
    
    return data


def _remove_buffer(data, buffer, axis=1):
    """
    Removes a temporal buffer (eg of zeros or additional samples) symmmetrically
    prepended/appended to data to avoid edge effects.

    data = _remove_buffer(data,buffer,axis=1)

    ARGS
    data    Data array where a buffer has been appended on both ends of time dimension.
            Can be any arbitrary size, typically (n_freqs,n_timepts+2*buffer,...).
    buffer  Int. Length (number of samples) of buffer appended to each end.
    axis    Int. Array axis to remove buffer from (ie time dim). Default: 1

    RETURNS
    data    Data array with buffer removed, reducing time axis to n_timepts
            (typically shape (n_freqs,n_timepts,...))
    """
    if axis == 1:
        return data[:,buffer:-buffer,...] # special case for default
    elif axis == 0:
        return data[buffer:-buffer,...]
    else:
        return (data.swapaxes(0,axis)[buffer:-buffer,...]
                    .swapaxes(axis,0))


# =============================================================================
# Other helper functions
# =============================================================================
def _infer_data_type(data):
    """ Infers type of data signal given -- 'raw' (real) | 'spectral' (complex) """
    if np.isrealobj(data):  return 'raw'
    else:                   return 'spectral'


def _iarange(start=0, stop=0, step=1):
    """
    Implements Numpy arange() with an inclusive endpoint. Same inputs as arange(), same
    output, except ends at stop, not stop - 1 (or more generally stop - step)

    r = _iarange(start=0,stop=0,step=1)
    
    Note: Must input all 3 arguments or use keywords (unlike flexible arg's in arange)    
    """
    if isinstance(step,int):    return np.arange(start,stop+1,step)
    else:                       return np.arange(start,stop+1e-12,step)


def _next_power_of_2(n):
    """
    Rounds x up to the next power of 2 (smallest power of 2 greater than n)
    """
    # todo  Think about switching this to use scipy.fftpack.next_fast_len
    return 1 if n == 0 else 2**ceil(log2(n))


def _infer_freq_scale(freqs):
    """ Determines if frequency sampling vector is linear, logarithmic, or uneven """
    # Determine if frequency scale is linear (all equally spaced)
    if np.allclose(np.diff(np.diff(freqs)),0):
        return 'linear'
    
    # Determine if frequency scale is logarithmic (all equally spaced in log domain)
    elif np.allclose(np.diff(np.diff(np.log2(freqs))),0):
        return 'log'
    
    # Otherwise assume arbitrary unevenly-sampled frequency axis (as in bandfilter/burst analysis)
    else:
        warn("Unable to determine scale of frequency sampling vector. Assuming it's arbitrary")
        return 'uneven'


def _freq_to_scale(freqs, wavelet='morlet', wavenumber=6):
    """
    Converts wavelet center frequency(s) to wavelet scales. Typically used to
    convert set of frequencies you want to set of scales, which pycwt understands.
    Currently only supports Morlet wavelets.

    scales = _freq_to_scale(freqs,mother,wavenumber)

    ARGS
    freqs       (n_freqs,) array-like. Set of desired wavelet frequencies

    wavelet     String. Type of wavelet. Default: 'morlet'

    wavenumber  Scalar. Wavelet wave number parameter ~ number of oscillations
                in each wavelet. Must be >= 6 to meet "admissibility constraint".
                Default: 6

    RETURNS
    scales      (n_freqs,) ndarray. Set of equivlent wavelet scales
    """
    freqs = np.asarray(freqs)
    if wavelet.lower() == 'morlet':
        return (wavenumber + sqrt(2 + wavenumber**2)) / (4*pi*freqs)
    else:
        raise ValueError("Unsupported value set for <wavelet>: '%s'" % wavelet)


def _interp1(x, y, xinterp, **kwargs):
    """
    Interpolates 1d data vector <y> sampled at index values <x> to
    new sampling vector <xinterp>
    Convenience wrapper around scipy.interpolate.interp1d w/o weird call structure
    """
    return interp1d(x,y,**kwargs).__call__(xinterp)


def _extract_triggered_data(data, smp_rate, event_times, window):
    """
    Extracts windowed chunks of data around given set of event times

    ARGS
    event_times (n_events,) array-like. List of times (s) of event triggers to
                extract data around. Times are referenced to 1st data sample (t=0)

    window      (2,) array-like. [start,end] of window (in s) to extract
                around each trigger.

    """
    # Convert event_times, window from s -> samples
    event_times = np.floor(np.asarray(event_times)*smp_rate).astype(int)
    window      = np.round(np.asarray(window)*smp_rate).astype(int)

    n_per_event = window[1] - window[0]
    n_events    = len(event_times)
    data_shape  = data.shape
    data_out    = np.zeros((n_per_event,n_events,*data_shape[1:]))

    for i_event,event in enumerate(event_times):
        idxs    = np.arange(event-window[0],event+window[1])
        data_out[:,i_event,...] = data[idxs,...]

    return data_out


def _check_window_lengths(windows,tol=1):
    """ 
    Ensures a set of windows are the same length. If not equal, but within given tolerance,
    windows are trimmed or expanded to the modal window length.
    
    ARGS
    windows (n_wins,2) array-like. Set of windows to test, given as series of [start,end].
    
    tol     Scalar. Max tolerance of difference of each window length from the modal value.
    
    RETURNS
    windows (n_wins,2) ndarray. Same windows, possibly slightly trimmed/expanded to uniform length
    """    
    windows = np.asarray(windows)
    
    window_lengths  = np.diff(windows,axis=1).squeeze()
    window_range    = np.ptp(window_lengths)
    
    # If all window lengths are the same, windows are OK and we are done here
    if np.allclose(window_lengths, window_lengths[0]): return windows
    
    # Compute mode of windows lengths and max difference from it
    modal_length    = mode(window_lengths)[0][0]    
    max_diff        = np.max(np.abs(window_lengths - modal_length))
    
    # If range is beyond our allowed tolerance, throw an error
    assert max_diff <= tol, \
        ValueError("Variable-length windows unsupported (range=%.1f). All windows must have same length" \
                    % window_range)
        
    # If range is between 0 and tolerance, we trim/expand windows to the modal length
    windows[:,1]    = windows[:,1] + (modal_length - window_lengths)
    return windows
    