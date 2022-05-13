# -*- coding: utf-8 -*-
"""
Analysis of oscillatory neural synchrony

Function list
-------------
Field-field synchrony
^^^^^^^^^^^^^^^^^^^^^
- synchrony :               General synchrony between pair of analog channels using given method
- coherence :               Time-frequency coherence between pair of analog channels
- ztransform_coherence  :   Z-transform coherence so ~ normally distributed
- plv :                     Phase locking value (PLV) between pair of analog channels
- ppc :                     Pairwise phase consistency (PPC) btwn pair of analog channels

Spike-field synchrony
^^^^^^^^^^^^^^^^^^^^^
- spike_field_coupling :    General spike-field coupling/synchrony btwn spike/LFP pair
- spike_field_coherence :   Spike-field coherence between a spike/LFP pair
- spike_field_plv :         Spike-field PLV between a spike/LFP pair
- spike_field_ppc :         Spike-field PPC between a spike/LFP pair

Data simulation
^^^^^^^^^^^^^^^
- simulate_multichannel_oscillation : Generates simulated oscillatory paired data

Function reference
------------------
"""
# Created on Thu Oct  4 15:28:15 2018
#
# @author: sbrincat
# TODO  Reformat jackknife functions to match randstats functions and move there?

from warnings import warn
import numpy as np

from spynal.utils import set_random_seed, setup_sliding_windows, index_axis, \
                         axis_index_slices
from spynal.spectra import spectrogram, simulate_oscillation
from spynal.randstats import jackknifes


# =============================================================================
# Field-Field Synchrony functions
# =============================================================================
def synchrony(data1, data2, axis=0, method='PPC', return_phase=False, single_trial=None,
              data_type=None, spec_method='wavelet', smp_rate=None, time_axis=None,
              taper_axis=None, keepdims=True, **kwargs):
    """
    Compute measure of synchrony between pair of channels of continuous (eg LFP)
    raw or spectral (time-frequency) data, using given estimation method

    Parameters
    ----------
    data1,data2 : ndarray, shape=(...,n_obs,...), dtype=float or complex
        Single-channel continuous (eg LFP) data for 2 distinct channels.
        Can be given as raw LFPs or complex-valued time-frequency transform.

        Trial/observation axis is assumed to be axis 0 unless given in `axis`.
        For raw data, axis corresponding to time must be given in `time_axis`.

        Other than those constraints, data can have arbitrary shape, with
        analysis performed in mass-bivariate fashion independently
        along each dimension other than observation `axis` (eg different
        frequencies, timepoints, conditions, etc.)

    axis : scalar, default: 0
        Axis corresponding to distinct observations/trials

    method : {'PPC','PLV','coherence'}, default: 'PPC'
        Synchrony estimation method:

        - 'PPC' : Pairwise Phase Consistency, measure of phase synchrony (see :func:`ppc`)
        - 'PLV' : Phase Locking Value, measure of phase synchrony (see :func:`plv`)
        - 'coherence' : coherence, measure of linear oscillatory coupling (see :func:`coherence`)

    return_phase : bool, default: False
        If False, only return measure of synchrony magnitude/strength between data1 and data2.
        If True, also returns mean phase difference (or coherence phase) between
        data1 and data2 (in radians) in additional output.

    single_trial : str or None, default: None
        What type of coherence estimator to compute:

        - None :       standard across-trial estimator
        - 'pseudo' :   single-trial estimates using jackknife pseudovalues
        - 'richter' :  single-trial estimates using actual jackknife estimates cf. Richter 2015

    data_type : {'raw','spectral'}, default: assume 'raw' if data is real; 'spectral' if complex
        What kind of data is input: 'raw' or 'spectral'?

    spec_method : {'wavelet','multitaper','bandfilter'}, default: 'wavelet'
        Method to use for spectral analysis (only used for *raw* data)

    smp_rate : scalar
        Sampling rate of data (only needed for *raw* data)

    time_axis : int
        Axis of data corresponding to time (only needed for *raw* data)

    taper_axis : int
        Axis of spectral data corresponding to tapers (only needed for *multitaper spectral* data)

    keepdims : bool, default: True
        If True, retains reduced trial and/or taper axes as length-one axes in output.
        If False, removes reduced trial,taper axes from outputs.

    **kwargs :
        All other kwargs passed as-is to specific synchrony estimation function.

    Returns
    -------
    sync : ndarray, shape=
        Synchrony magnitude/strength (PPC/PLV/coherence) between data1 and data2.
        If data is spectral, this has same shape as data, but with `axis` reduced
        to a singleton or removed, depending on value of `keepdims`.
        If data is raw, this has same shape as data with `axis` reduced or removed
        and a new frequency axis inserted immediately before `time_axis`.

    freqs : ndarray, shape=(n_freqs,)
        List of frequencies in `sync`. Only returned for raw data, [] otherwise.

    timepts : ndarray, shape=(n_timepts_out,)
        List of timepoints in `sync` (in s, referenced to start of data).
        Only returned for raw data, [] otherwise.

    dphi : ndarray, optional
        Mean phase difference (or coherence phase) between data1 and data2 in radians.
        Same shape as `sync`.
        Positive values correspond to data1 leading data2.
        Negative values correspond to data1 lagging behind data2.
        Optional: Only returned if return_phase is True.

    Examples
    --------
    sync, freqs, timepts = synchrony(data1, data2, axis=0, method='PPC', return_phase=False)

    sync, freqs, timepts, dphi = synchrony(data1, data2, axis=0, method='PPC', return_phase=True)

    References
    ----------
    - Single-trial method:    Womelsdorf et al (2006) https://doi.org/10.1038/nature04258
    - Single-trial method:    Richter et al (2015) https://doi.org/10.1016/j.neuroimage.2015.04.040
    """
    method = method.lower()
    if method in ['ppc','pairwise_phase_consistency']:  sync_func = ppc
    elif method in ['plv','phase_locking_value']:       sync_func = plv
    elif method in ['coh','coherence']:                 sync_func = coherence
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)

    return sync_func(data1, data2, axis=axis, return_phase=return_phase, single_trial=single_trial,
                     data_type=data_type, spec_method=spec_method, smp_rate=smp_rate,
                     time_axis=time_axis, taper_axis=taper_axis, keepdims=keepdims, **kwargs)


def coherence(data1, data2, axis=0, return_phase=False, transform=None, single_trial=None,
              spec_method='wavelet', data_type=None, smp_rate=None, time_axis=None,
              taper_axis=None, keepdims=True, **kwargs):
    """
    Compute coherence between pair of channels of raw or spectral (time-frequency) data
    (LFP or spikes)

    Coherence is a spectral analog of linear correlation that takes both phase and amplitude
    into account.

    Only parameters differing from :func:`synchrony` are described here.

    Parameters
    ----------
    transform : 'Z' or None, default: None
        Transform to apply to all computed coherence values.
        Set=None to return raw, untransformed coherence.
        Set='Z' to Z-transform coherence using Jarvis & Mitra (2001) method.

    **kwargs :
        Any other keyword args passed as-is to spectrogram() function.
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")

    assert (single_trial is None) or (single_trial in ['pseudo','richter']), \
        ValueError("Parameter <single_trial> must be = None, 'pseudo', or 'richter'")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    # Check if raw data input. If so, compute spectral transform first
    data1, data2, freqs, timepts, axis, time_axis, taper_axis = \
        _sync_raw_to_spectral(data1, data2, smp_rate, axis, time_axis, taper_axis,
                              spec_method, data_type, **kwargs)

    # For multitaper, compute means across trials, tapers; degrees of freedom = 2*n_trials*n_tapers
    if spec_method == 'multitaper':
        reduce_axes = (axis,taper_axis)
        df = 2*np.prod([data1.shape[ax] for ax in reduce_axes])
    # Otherwise just do means across trials; df = 2*n_trials
    else:
        reduce_axes = axis
        df = 2*data1.shape[axis]

    # Setup actual function call for any transform to compute on raw coherence values
    if (transform is None) or callable(transform):
        transform_ = transform
    elif transform.lower() in ['z','ztransform']:
        transform_ = lambda coh: ztransform_coherence(coh, df)
    else:
        raise ValueError("Unsupported value '%s' set for <transform>" % transform)

    # Compute cross-spectrum and auto-spectrum of each channel
    auto_spec1 = data1*data1.conj()
    auto_spec2 = data2*data2.conj()
    cross_spec = data1*data2.conj()

    def _cross_auto_to_coh(cross_spec, auto_spec1, auto_spec2, axis,
                           return_phase, transform, keepdims):
        """ Compute coherence from cross-spectrum and pair of auto-spectra """
        # Average spectra across observatations (trials and/or tapers)
        # Note: .real deals with floating point error, converts complex dtypes to float
        auto_spec1 = np.mean(auto_spec1, axis=axis, keepdims=keepdims).real
        auto_spec2 = np.mean(auto_spec2, axis=axis, keepdims=keepdims).real
        cross_spec = np.mean(cross_spec, axis=axis, keepdims=keepdims)

        # Calculate complex coherency as cross-spectrum / product of square root of spectra
        coherency = cross_spec / np.sqrt(auto_spec1*auto_spec2)

        # Convert complex coherency -> coherence (magnitude of coherency)
        coherence = np.abs(coherency).real

        # Perform any requested tranform on coherence (eg z-scoring)
        if transform is not None: coherence = transform(coherence)

        # Optionally also extract coherence phase angle
        if return_phase:    return coherence, np.angle(coherency)
        else:               return coherence, None


    # Standard across-trial coherence estimator
    if single_trial is None:
        coh, dphi = _cross_auto_to_coh(cross_spec, auto_spec1, auto_spec2, reduce_axes,
                                        return_phase, transform_, keepdims)

    # Single-trial coherence estimator using jackknife resampling method
    else:
        # Jackknife resampling of coherence statistic (this is the 'richter' estimator)
        # Note: Allow for reduction along taper axis within resampled stat function, but only
        #       resample across trial axis--want sync estimates for single-trials, not tapers
        jackfunc = lambda s12,s1,s2: _cross_auto_to_coh(s12, s1, s2, reduce_axes,
                                                        False, transform_, True)[0]
        coh_shape = list(cross_spec.shape)
        if spec_method == 'multitaper': coh_shape[taper_axis] = 1
        n_jack = coh_shape[axis]
        ndim = len(coh_shape)

        # Create generator with n length-n vectors, each of which excludes 1 trial
        resamples = jackknifes(n_jack)

        # Do jackknife resampling -- estimate statistic w/ each observation left out
        coh = np.empty(coh_shape, dtype=float)
        for trial,sel in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            slices_in   = axis_index_slices(axis, sel, ndim)
            slices_out  = axis_index_slices(axis, [trial], ndim)
            coh[slices_out] = jackfunc(cross_spec[slices_in],
                                       auto_spec1[slices_in], auto_spec2[slices_in])

        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        # Note: n here is the number of jackknifes computed = n_trials
        if single_trial == 'pseudo':
            coh_full = jackfunc(cross_spec, auto_spec1, auto_spec2)
            coh = jackknife_to_pseudoval(coh_full, coh, data1.shape[axis])

        if not keepdims and (spec_method == 'multitaper'): coh = coh.squeeze(axis=taper_axis)

    if return_phase:    return coh, freqs, timepts, dphi
    else:               return coh, freqs, timepts


def ztransform_coherence(coh, df, beta=23/20):
    """
    z-transform coherence values to render them approximately normally distributed

    Parameters
    ----------
    coh : ndarray, shape=Any
        Raw coherence values

    df : int
        Degrees of freedom of coherence estimates.
        For multitaper spectral estimates, this is usually df = 2*n_trials*n_tapers.
        For other estimates, this is usually df = 2*n_trials.

    beta : scalar, default: 23/20
        Mysterious number from Jarvis & Mitra to make output z-scores approximately normal

    Returns
    -------
    z : ndarray, shape=Any
        z-transformed coherence. Same shape as `coh`.

    References
    ----------
    - Jarvis & Mitra (2001) Neural Computation https://doi.org/10.1162/089976601300014312
    - Hipp, Engel, Siegel (2011) Neuron https://doi.org/10.1016/j.neuron.2010.12.027
    """
    return beta*(np.sqrt(-(df-2)*np.log(1-coh**2)) - beta)


def plv(data1, data2, axis=0, return_phase=False, transform=None, single_trial=None,
        spec_method='wavelet', data_type=None, smp_rate=None,
        time_axis=None, taper_axis=None, keepdims=True, **kwargs):
    """
    Compute phase locking value (PLV) between raw or spectral (time-frequency) LFP data.

    PLV is a measure of phase synchrony that ignores signals amplitudes.

    PLV is the mean resultant length (magnitude of the vector mean) of phase
    differences dphi btwn phases of data1 and data2:

        dphi = phase(data1) - phase(data2)
        PLV  = abs( trial_mean(exp(i*dphi)) )

    Only parameters differing from :func:`synchrony` are described here.

    Parameters
    ----------
    transform : 'PPC' or None, default: None
        Transform to apply to all computed PLV values.
        Set=None to return untransformed PLV.
        Set='PPC' to transform to debiased estimator of PLV^2 (aka Pairwise Phase Consistency/PPC).

    **kwargs :
        Any other keyword args passed as-is to spectrogram() function.

    References
    ----------
    Lachaux (1999) https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C
    """
    assert data1.shape[axis] == data2.shape[axis], \
        ValueError("data1,data2 must have same number of observations (trials)")
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis
    if (taper_axis is not None) and (taper_axis < 0): taper_axis = data1.ndim + taper_axis

    # Check if raw data input. If so, compute spectral transform first
    data1, data2, freqs, timepts, axis, time_axis, taper_axis = \
        _sync_raw_to_spectral(data1, data2, smp_rate, axis, time_axis, taper_axis,
                              spec_method, data_type, **kwargs)

    # For multitaper, compute means across trials, tapers; otherwise just do means across trials
    reduce_axes = (axis,taper_axis) if spec_method == 'multitaper' else axis
    n = np.prod([data1.shape[ax] for ax in reduce_axes]) if spec_method == 'multitaper' else \
        data1.shape[axis]

    # Setup actual function call for any transform to compute on raw PLV values
    if (transform is None) or callable(transform):
        transform_ = transform
    else:
        if transform.lower() == 'ppc':
            transform_ = lambda PLV: plv_to_ppc(PLV, n)
        else:
            raise ValueError("Unsupported value '%s' set for <transform>" % transform)

    # Compute normalized cross-spectrum btwn the two channels
    # Cross-spectrum-based method adapted from FieldTrip ft_conectivity_ppc()
    # Note: Traditional circular mean-based algorithm is ~3x slower
    cross_spec = data1*data2.conj()                 # Compute cross-spectrum
    cross_spec = cross_spec / np.abs(cross_spec)    # Normalize cross-spectrum


    def _cross_to_plv(cross_spec, axis, return_phase, transform, keepdims):
        """ Compute PLV from cross-spectrum """
        # Vector-average complex spectra across observatations (trials and/or tapers)
        vector_mean = np.mean(cross_spec, axis=axis, keepdims=keepdims)

        # Compute PLV = magnitude of complex vector mean = mean resultant length
        # Note: .real deals with floating point error, converts complex dtypes to float
        PLV = np.abs(vector_mean).real

        # Perform any requested tranform on PLV (eg convert to PPC)
        if transform is not None: PLV = transform(PLV)

        # Optionally also extract mean relative phase angle
        if return_phase:    return PLV, np.angle(vector_mean)
        else:               return PLV, None


    # Standard across-trial PLV estimator
    if single_trial is None:
        PLV, dphi = _cross_to_plv(cross_spec, reduce_axes, return_phase, transform_, keepdims)

    # Single-trial PLV estimator using jackknife resampling method
    else:
        # Jackknife resampling of PLV statistic (this is the 'richter' estimator)
        # Note: Allow for reduction along taper axis within resampled stat function, but only
        #       resample across trial axis--want sync estimates for single-trials, not tapers
        jackfunc = lambda s12: _cross_to_plv(s12, reduce_axes, False, transform_, True)[0]
        plv_shape = list(data1.shape)
        if spec_method == 'multitaper': plv_shape[taper_axis] = 1
        n_jack = plv_shape[axis]
        ndim = len(plv_shape)

        # Create generator with n length-n vectors, each of which excludes 1 trial
        resamples = jackknifes(n_jack)

        # Do jackknife resampling -- estimate statistic w/ each observation left out
        PLV = np.empty(plv_shape, dtype=float)
        for trial,sel in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            slices_in   = axis_index_slices(axis, sel, ndim)
            slices_out  = axis_index_slices(axis, [trial], ndim)
            PLV[slices_out] = jackfunc(cross_spec[slices_in])

        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        # Note: n here is the number of jackknifes computed = n_trials
        if single_trial == 'pseudo':
            plv_full = jackfunc(cross_spec)
            PLV = jackknife_to_pseudoval(plv_full, PLV, data1.shape[axis])

        if not keepdims and (spec_method == 'multitaper'): PLV = PLV.squeeze(axis=taper_axis)

    if return_phase:    return  PLV, freqs, timepts, dphi
    else:               return  PLV, freqs, timepts

# Alias function with full name
phase_locking_value = plv
""" Alias of :func:`plv`. See there for details. """


def ppc(data1, data2, axis=0, return_phase=False, single_trial=None,
        spec_method='wavelet', data_type=None, smp_rate=None,
        time_axis=None, taper_axis=None, keepdims=True, **kwargs):
    """
    Compute pairwise phase consistency (PPC) between raw or spectral
    (time-frequency) LFP data, which is bias-corrected (unlike PLV and coherence,
    which are biased by n)

    PPC is a measure of phase synchrony that ignores signal amplitude.

    PPC computes the cosine of the absolute angular distance (the vector dot product)
    for all given pairs of relative phases, i.e., it computes how similar the relative
    phase observed in one trial is to the relative phase observed in another trial

    PPC is also an debiased estimator of the square of the mean resultant length (PLV^2),
    and can be expressed (and computed efficiently) in terms of PLV and n:

        PPC = (n*PLV^2 - 1) / (n-1)

    Only parameters differing from :func:`synchrony` are described here.

    Parameters
    ----------
    **kwargs :
        Any other keyword args passed as-is to spectrogram() function.

    References
    ----------
    - Original concept:   Vinck et al. (2010) https://doi.org/10.1016/j.neuroimage.2010.01.073
    - Relation to PLV:    Kornblith, Buschman, Miller (2015) https://doi.org/10.1093/cercor/bhv182
    """
    # Simply call plv() with a 'PPC' transform
    return plv(data1, data2, axis=axis, return_phase=return_phase, transform='PPC',
               single_trial=single_trial, spec_method=spec_method, data_type=data_type,
               smp_rate=smp_rate,  time_axis=time_axis, taper_axis=taper_axis,
               keepdims=keepdims, **kwargs)

# Alias function with full name
pairwise_phase_consistency = ppc
""" Alias of :func:`ppc`. See there for details. """


def plv_to_ppc(PLV, n):
    """ Convert PLV to PPC as PPC = (n*PLV^2 - 1)/(n-1) """
    return (n*PLV**2 - 1) / (n - 1)


# =============================================================================
# Spike-Field Synchrony functions
# =============================================================================
def spike_field_coupling(spkdata, lfpdata, axis=0, method='PPC', return_phase=False,
                         time_axis=None, taper_axis=None, timepts=None,
                         data_type=None, spec_method='multitaper', smp_rate=None,
                         keepdims=True, **kwargs):
    """
    Computs measure of pairwise coupling between a pair of spike and continuous (eg LFP)
    raw or spectral (time-frequency) data, using given estimation method

    Parameters
    ----------
    spkdata : ndarray, shape=(...,n_obs,...), dtype=bool
        Single-channel binary spike trains (with 1's labelling spike times, 0's otherwise).

        For coherence: Can be given either as raw binary spike trains or as their spectral
        transform, but must have same data type (raw or spectral) and shape as `lfpdata`.

        For PLV/PPC: Shape is arbitrary, but MUST have same shape as `lfpdata`
        for raw `lfpdata`, and same dimensionality as `lfpdata` for spectral `lfpdata`.
        Thus, if `lfpdata` is spectral, must pre-pend singleton (length-1) axis to
        spkdata to match (eg using np.newaxis).

    lfpdata : ndarray, shape=(...,n_obs,...)
        Single-channel continuous (eg LFP) data.
        Can be given as raw LFPs or complex-valued time-frequency transform.

        Trial/observation axis is assumed to be axis 0 unless given in `axis`.
        For raw data, axis corresponding to time must be given in `time_axis`.

        Other than those constraints, data can have arbitrary shape, with analysis performed
        in mass-bivariate fashion independently along each axis other than observation `axis`
        (eg different frequencies, timepoints, conditions, etc.)

    axis : int, default: 0
        Axis corresponding to distinct observations/trials

    method : {'PPC','PLV','coherence'}, default: 'PPC'
        Spike-field coupling estimation method

        - 'PPC' : Pairwise Phase Consistency (see :func:`spike_field_ppc`)
        - 'PLV' : Phase Locking Value (see :func:`spike_field_plv`)
        - 'coherence' : coherence (see :func:`spike_field_coherence`)

    return_phase : bool, default: False
        If False, only return measure of synchrony magnitude/strength between data1 and data2.
        If True, also returns mean LFP phase of spikes (or coherence phase) in additional output.

    time_axis : int
        Axis of data corresponding to time. Required input for phase-based methods (PLV/PPC).
        For coherence, only needed for `data_type` = 'raw' AND `spec_method` = 'multitaper'.

    taper_axis : int
        Axis of spectral data corresponding to tapers. Only needed for multitaper spectral data.

    timepts : array-like, shape=(n_timepts,), Default: (0 - n_timepts-1)/smp_rate
        Time sampling vector for data. Default value starts at 0, with spacing = 1/smp_rate.
        For phase methods, should be in same time units as `width`/`spacing`/`lims` or `timewins`.

    data_type : {'raw','spectral'}, default: assume 'raw' if data is real; 'spectral' if complex
        What kind of data is input: 'raw' or 'spectral'?

    spec_method : {'wavelet','multitaper','bandfilter'}, default: 'wavelet'
        Method to use for (or already used for) spectral analysis.
        NOTE: Must be input for multitaper spectral data, so taper axis is handled appropriately.
        Otherwise, only used for *raw* data.

    smp_rate : scalar
        Sampling rate of data (only needed for *raw* data)

    keepdims : bool, default: True
        If True, retains reduced trial and/or taper axes as length-one axes in output.
        If False, removes reduced trial,taper axes from outputs.

    **kwargs :
        All other kwargs passed as-is to specific synchrony estimation function.

    Returns
    -------
    sync : ndarray
        Magnitude/strength of spike-field coupling (coherence or PLV/PPC magnitude).

        For phase-based methods, time windows without any spikes are set = np.nan.

        If data is spectral, this has same shape as data, but with `axis` removed.
        If data is raw, this has same shape with `axis` removed and a new
        frequency axis inserted immediately before `time_axis`.

        TODO
        If lfpdata is spectral, this has same shape, but with `axis` removed
        (and taper_axis as well for multitaper), and time axis reduced to n_timewins.
        If lfpdata is raw, this has same shape with `axis` removed, `time_axis`
        reduced to n_timewins, and a new frequency axis inserted immediately
        before `time_axis`.

    freqs : ndarray, shape=(n_freqs,)
        List of frequencies in `sync`. Only returned for raw data, [] otherwise.

    timepts : ndarray, shape=(n_timepts,)
        List of timepoints in `sync` (in s, referenced to start of data).
        Only returned for raw data, [] otherwise.

    n : ndarray, shape=(n_timepts,), dtype=int
        Number of spikes contributing to synchrony computations.
        Value only returned for phase-based measures (PLV/PPC), for coherence, returns None.

    phi : ndarray, optional
        Mean phase of LFPs at spike times (or phase of complex coherency) in radians.
        Optional: Only returned if return_phase is True.

    Examples
    --------
    sync,freqs,timepts,n = spike_field_coupling(spkdata,lfpdata,return_phase=False)

    sync,freqs,timepts,n,phi = spike_field_coupling(spkdata,lfpdata,return_phase=True)
    """
    method = method.lower()
    if method in ['ppc','pairwise_phase_consistency']:  sfc_func = spike_field_ppc
    elif method in ['plv','phase_locking_value']:       sfc_func = spike_field_plv
    elif method in ['coh','coherence']:                 sfc_func = spike_field_coherence
    else:
        raise ValueError("Unsupported value '%s' given for <method>. \
                         Should be 'PPC'|'PLV'|'coherence'" % method)

    return sfc_func(spkdata, lfpdata, axis=axis, time_axis=time_axis, taper_axis=taper_axis,
                    timepts=timepts, data_type=data_type, spec_method=spec_method,
                    smp_rate=smp_rate, return_phase=return_phase, keepdims=keepdims, **kwargs)


def spike_field_coherence(spkdata, lfpdata, axis=0, time_axis=None, taper_axis=None,
                          timepts=None, transform=None,
                          data_type=None, spec_method='multitaper', smp_rate=None,
                          return_phase=False, keepdims=True, **kwargs):
    """
    Compute pairwise coherence between single-channel spiking data and LFP data

    Only parameters differing from :func:`spike_field_coupling` are described here.

    Parameters
    ----------
    transform : 'Z' or None, default: None
        Transform to apply to all computed coherence values.
        Set=None to return raw, untransformed coherence.
        Set='Z' to Z-transform coherence using Jarvis & Mitra (2001) method.

    **kwargs :
        Any other keyword args passed as-is to spectrogram() function
    """
    if axis < 0: axis = lfpdata.ndim + axis
    if time_axis < 0: time_axis = lfpdata.ndim + time_axis

    # Check if raw data input. If so, compute spectral transform first
    spkdata, lfpdata, freqs, timepts, smp_rate, axis, time_axis, taper_axis = \
        _sfc_raw_to_spectral(spkdata, lfpdata, smp_rate, axis, time_axis, taper_axis, timepts,
                             'coherence', spec_method, data_type, **kwargs)

    extra_args = dict(axis=axis, data_type='spectral', spec_method=spec_method,
                      transform=transform, return_phase=return_phase)
    if spec_method == 'multitaper': extra_args.update(taper_axis=taper_axis)

    if return_phase:
        coh,_,_,phi = coherence(spkdata, lfpdata, keepdims=keepdims, **extra_args)
        return coh,freqs,timepts,None,phi
    else:
        coh,_,_ = coherence(spkdata, lfpdata, keepdims=keepdims, **extra_args)
        return coh,freqs,timepts,None


def spike_field_plv(spkdata, lfpdata, axis=0, time_axis=None, taper_axis=None,
                    timepts=None, width=0.5, spacing=None, lims=None, timewins=None,
                    data_type=None, spec_method='wavelet', smp_rate=None,
                    return_phase=False, keepdims=True, **kwargs):
    """
    Compute phase locking value (PLV) of spike-triggered LFP phase

    PLV is the mean resultant length (magnitude of the vector mean) of the
    spike-triggered LFP phase 'phi'::

        PLV  = abs( trial_mean(exp(i*phi)) )

    Because spiking response are sparse, spike-LFP PLV is typically computed within sliding
    time windows (ie summation across trials AND across within-window timepoints). These can
    be specified either explicitly using `timewins` or implicitly using `width`/`spacing`/`lims`.

    Parameters
    ----------
    width : scalar, default: 0.5 (500 ms)
        Width (in s) of sliding time windows for computing PLV

    spacing : scalar, default: set = `width` (ie exactly non-overlapping windows)
        Spacing (in s) of sliding time windows for computing PLV

    lims : array-like, shape=(2,), default: (timepts[0],timepts[-1]) (full sampled time of data)
        [Start,end] time limits (in s) for full series of sliding windows

    timewins : ndarray, shape=(n_timewins,2), Default: setup_sliding_windows(width,lims,spacing)
        Alternative method for setting sliding time windows; overrides `width`/`spacing`/`lims`.
        Custom time windows to compute PLV within, given as explicit series of window [start,end]'s
        (in s). Can have any arbitrary width/spacing (eg to compute PLV in arbitary time epochs).
        Default generates sliding windows with `width` and `spacing` from `lims[0]` to `lims[1]`

        Special case: Set = (lim[0],lim[1]) to compute PLV *spectrum* over entire data time period.

    **kwargs :
        Any other keyword args passed as-is to spectrogram() function.

    References
    ----------
    Lachaux (1999) https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C
    """
    max_axis_mismatch = 2 if spec_method == 'multitaper' else 1
    assert (spkdata.ndim == lfpdata.ndim) and \
           ((np.array(spkdata.shape) != np.array(lfpdata.shape)).sum() <= max_axis_mismatch), \
        ValueError("Spiking data " + str(spkdata.shape) +
                   " must have same size/shape as LFP data " + str(lfpdata.shape) +
                   " (w/ singleton to match freq [and taper] axis)")
    if (timepts is None) and ((timewins is not None) or (lims is not None)):
        assert smp_rate is not None, \
            ValueError("If no value is input for <timepts>, must input value for <smp_rate>")
        warn("No value input for <timepts>. Setting = (0 - n_timepts-1)/smp_rate.\n"
             "Assuming <lims> and/or <timewins> are given in same timebase.")

    # if data_type is None: data_type = _infer_data_type(lfpdata)
    if axis < 0: axis = lfpdata.ndim + axis
    if time_axis < 0: time_axis = lfpdata.ndim + time_axis

    # Default timepts to range from 0 - n_timepts/smp_rate
    if timepts is None:     timepts = np.arange(lfpdata.shape[time_axis]) / smp_rate
    elif smp_rate is None:  smp_rate = 1 / np.diff(timepts).mean()

    # Are we computing PLV as a first step for PPC? (should only be set when called from ppc())
    _for_ppc = kwargs.pop('_for_ppc', False)

    # Check if raw data input. If so, compute spectral transform first
    spkdata, lfpdata, freqs, timepts, smp_rate, axis, time_axis, taper_axis = \
        _sfc_raw_to_spectral(spkdata, lfpdata, smp_rate, axis, time_axis, taper_axis, timepts,
                             'plv', spec_method, data_type, **kwargs)

    # Default lims to [start,end] of timepts
    # (Note: do this after multitaper timepts adjustment above)
    if lims is None: lims = (timepts[0],timepts[-1])

    # For multitaper spectral data, reshape lfpdata s.t. tapers and trials are on same axis
    if spec_method == 'multitaper':
        assert taper_axis is not None, \
            ValueError("For multitaper spec_method, must input a value for taper_axis")

        n_tapers = lfpdata.shape[taper_axis]

        # STOPPED HERE Need to figure out how to undo taper axis folding after PLV computation
        # OR keep taper axis thru PLV computation?

        # Move taper axis next to trial axis
        lfpdata = np.moveaxis(lfpdata,taper_axis,axis)
        # If trial axis was after taper axis, taper axis is now after trial, so unwrap in F order
        # If trial axis was before taper axis, taper axis is now before trial, so unwrap in C order
        order = 'F' if axis > taper_axis else 'C'
        axis_ = axis - 1 if axis > taper_axis else axis
        # Reshape lfpdata so tapers on same axis as trials -> (...,n_trials*n_tapers,...)
        lfpdata = lfpdata.reshape((*lfpdata.shape[0:axis_], -1, *lfpdata.shape[(axis_+2):]),
                                  order=order)

        # Expand spkdata trial axis to n_trials*n_tapers to match lfpdata and remove taper axis
        tiler   = np.ones((spkdata.ndim,),dtype=int)
        tiler[axis] = n_tapers
        spkdata = np.tile(spkdata, tuple(tiler))
        spkdata = spkdata.squeeze(axis=taper_axis)

        # Adjust axes for removal of taper axis
        if time_axis > taper_axis: time_axis -= 1
        if axis > taper_axis: axis -= 1

    data_ndim = lfpdata.ndim
    # Move time and trials/observations axes to end of data arrays -> (...,n_timepts,n_trials)
    if not ((time_axis == data_ndim-2) and (axis == data_ndim-1)):
        lfpdata = np.moveaxis(lfpdata,time_axis,-1)
        lfpdata = np.moveaxis(lfpdata,axis,-1)
        spkdata = np.moveaxis(spkdata,time_axis,-1)
        spkdata = np.moveaxis(spkdata,axis,-1)
    data_shape = lfpdata.shape   # Cache data shape after axes shift
    n_timepts,n_trials = data_shape[-2], data_shape[-1]

    # Unwrap all other axes (incl. frequency) -> (n_data_series,n_timepts,n_trials)
    if data_ndim > 2:
        lfpdata = np.reshape(lfpdata, (-1,n_timepts,n_trials))
        spkdata = np.reshape(spkdata, (-1,n_timepts,n_trials))

    # Normalize LFP spectrum/spectrogram so data is all unit-length complex vectors
    lfpdata = lfpdata / np.abs(lfpdata)

    # Set timewins based on given parameters if not set explicitly in args
    if timewins is None:
        timewins = setup_sliding_windows(width,lims,spacing)
    else:
        timewins = np.asarray(timewins)
        width = np.diff(timewins,axis=1).mean()

    # Convert time sampling vector and time windows to int-valued ms,
    #  to avoid floating-point issues in indexing below
    timepts_ms  = np.round(timepts*1000).astype(int)
    timewins_ms = np.round(timewins*1000).astype(int)

    n_data_series = lfpdata.shape[0]
    n_timepts_out = timewins.shape[0]

    vector_mean = np.full((n_data_series,n_timepts_out,1),np.nan,dtype=complex)

    n = np.zeros((n_timepts_out,),dtype=int)

    # Are we computing PLV within temporal windows or at each timepoint
    # (If we don't know smp_rate, assume windowed. Same result, but slower algorithm.)
    do_timewins = (smp_rate is None) or not np.isclose(width, 1/smp_rate)

    # Compute PLV by vector averaging over trials and within given sliding time windows
    if do_timewins:
        for i_win,timewin in enumerate(timewins_ms):
            # Boolean vector flagging all time points within given time window
            tbool = (timepts_ms >= timewin[0]) & (timepts_ms <= timewin[1])

            # Logical AND btwn window and spike train booleans to get spikes in window
            win_spikes = spkdata & tbool[np.newaxis,:,np.newaxis]

            # Count of all spikes within time window across all trials/observations
            n[i_win] = win_spikes.sum()

            # If no spikes in window, can't compute PLV. Skip and leave = nan.
            if n[i_win] == 0: continue

            # Use windowed spike times to index into LFPs and compute complex mean
            # across all spikes (within all trials/observations and window timepoints)
            vector_mean[:,i_win,0] = \
                (lfpdata[np.tile(win_spikes,(n_data_series,1,1))].reshape((n_data_series,n[i_win]))
                                                                 .mean(axis=-1))

            # todo Need to timetest against these alternatives
            # vector_mean[:,i_win] = lfpdata[win_spikes[[0]*n_data_series,:,:]].mean(axis=(-1,-2))
            # vector_mean[:,i_win] = lfpdata[win_spikes[np.zeros((n_data_series,),dtype=int),:,:]].mean(axis=(-1,-2))

    # Compute PLV by vector averaging over trials at each individual timepoint
    else:
        for i_time in range(n_timepts):
            # Count of all spikes within time window across all trials/observations
            n[i_time] = spkdata[0,i_time,:].sum()

            # If no spikes in window, can't compute PLV. Skip and leave = nan.
            if n[i_time] == 0: continue

            # Use spike times to index into LFPs and compute complex mean
            # across all spikes (within all trials/observations)
            vector_mean[:,i_time,0] = lfpdata[:,i_time,spkdata[0,i_time,:]].mean(axis=-1)

    # Reshape axes (incl. frequency) to original data shape
    if data_ndim > 2:
        vector_mean = np.reshape(vector_mean, (*data_shape[:-2],n_timepts_out,1))
    # Move time and trials/observations axes to end of data arrays -> (...,n_timepts,1)
    if not ((time_axis == data_ndim-2) and (axis == data_ndim-1)):
        vector_mean = np.moveaxis(vector_mean,-1,axis)
        vector_mean = np.moveaxis(vector_mean,-1,time_axis)

    # HACK If computing PLV as a first step for PPC, temporarily expand dimensionality of n
    # so it will broadcast against PLV (it gets squeezed back down to (n_timewins,) in ppc())
    if _for_ppc:
        slicer = [np.newaxis]*vector_mean.ndim  # Create (PLV.ndim,) list of np.newaxis's
        slicer[time_axis] = slice(None)         # Set <time_axis> element to slice as if set=':'
        if not keepdims: del slicer[axis]       # Remove trial axis if not keeping singleton
        n = n[tuple(slicer)]                    # Expand n to match dimensionality of PLV

    if not keepdims: vector_mean = vector_mean.squeeze(axis=axis)

    # Compute absolute value of complex vector mean = mean resultant = PLV
    # and optionally the mean phase angle as well. Also return spike counts.
    if return_phase:
        return np.abs(vector_mean), freqs, timewins.mean(axis=1), n, np.angle(vector_mean)
    else:
        return np.abs(vector_mean), freqs, timewins.mean(axis=1), n


# Alias function with full name
spike_field_phase_locking_value = spike_field_plv
""" Alias of :func:`spike_field_plv`. See there for details. """


def spike_field_ppc(spkdata, lfpdata, axis=0, time_axis=None, taper_axis=None,
                    timepts=None, width=0.5, spacing=None, lims=None, timewins=None,
                    data_type=None, spec_method='wavelet', smp_rate=None,
                    return_phase=False, keepdims=True, **kwargs):
    """
    Compute pairwise phase consistency (PPC) of spike-triggered LFP phase,
    which is bias-corrected (unlike PLV and coherence, which are biased by n)

    PPC is an debiased estimator of PLV^2, and can be expressed (and computed
    efficiently) in terms of PLV and n::

        PPC = (n*PLV^2 - 1) / (n-1)

    Because spiking response are sparse, spike-LFP PPC is typically computed within sliding
    time windows (ie summation across trials AND across within-window timepoints). These can
    be specified either explicitly using `timewins` or implicitly using `width`/`spacing`/`lims`.

    Parameters
    ----------
    width : scalar, default: 0.5 (500 ms)
        Width (in s) of sliding time windows for computing PPC

    spacing : scalar, default: set = `width` (ie exactly non-overlapping windows)
        Spacing (in s) of sliding time windows for computing PPC

    lims : array-like, shape=(2,), default: (timepts[0],timepts[-1]) (full sampled time of data)
        [Start,end] time limits (in s) for full series of sliding windows

    timewins : ndarray, shape=(n_timewins,2), Default: setup_sliding_windows(width,lims,spacing)
        Alternative method for setting sliding time windows; overrides `width`/`spacing`/`lims`.
        Custom time windows to compute PPC within, given as explicit series of window [start,end]'s
        (in s). Can have any arbitrary width/spacing (eg to compute PPC in arbitary time epochs).
        Default generates sliding windows with `width` and `spacing` from `lims[0]` to `lims[1]`

        Special case: Set = (lim[0],lim[1]) to compute PPC *spectrum* over entire data time period.

    **kwargs :
        Any other keyword args passed as-is to spectrogram() function.

    References
    ----------
    - Original concept:   Vinck et al. (2010) https://doi.org/10.1016/j.neuroimage.2010.01.073
    - Relation to PLV:    Kornblith, Buschman, Miller (2015) https://doi.org/10.1093/cercor/bhv182
    """
    extra_args = dict(axis=axis, time_axis=time_axis, taper_axis=taper_axis, timepts=timepts,
                      width=width, spacing=spacing, lims=lims, timewins=timewins,
                      data_type=data_type, spec_method=spec_method, smp_rate=smp_rate,
                      return_phase=return_phase, keepdims=keepdims, _for_ppc=True, **kwargs)

    if return_phase:
        PLV,freqs,timepts,n,phi = spike_field_plv(spkdata, lfpdata, **extra_args)
        return plv_to_ppc(PLV,n), freqs, timepts, n.squeeze(), phi

    else:
        PLV,freqs,timepts,n = spike_field_plv(spkdata, lfpdata, **extra_args)
        return plv_to_ppc(PLV,n), freqs, timepts, n.squeeze()

# Alias function with full name
spike_field_pairwise_phase_consistency = spike_field_ppc
""" Alias of :func:`spike_field_ppc`. See there for details. """


# =============================================================================
# Utility functions for generating single-trial jackknife pseudovalues
# =============================================================================
def two_sample_jackknife(func, data1, data2, *args, axis=0, shape=None, dtype=None, **kwargs):
    """
    Jackknife resampling of arbitrary statistic computed by `func` on `data1` and `data2`

    Parameters
    ----------
    func : callable
        Takes data1,data2 (and any other args) as input, returns statistic to be jackknifed

    data1,data2 : ndarray, shape:Any
        Data to compute jackknife resamples of. Shape is arbitrary but must be same.

    axis : int, default: 0
        Observation (trial) axis of `data1` and `data2`

    shape : tuple, shape=(ndims,), dtype=int, default: data1.shape
        Shape for output array `stat`. Default to same as input.

    dtype : str, default: data1.dtype
        dtype for output array `stat`. Default to same as input.

    *args, **kwargs :
        Any additional arguments passed directly to `func`

    Returns
    -------
    stat : ndarray, shape=Any
        Jackknife resamples of statistic. Same size as data1,data2.
    """
    if shape is None: shape = data1.shape
    if dtype is None: dtype = data1.dtype

    n = data1.shape[axis]
    ndim = data1.ndim

    # Create generator with n length-n vectors, each of which excludes 1 trial
    resamples = jackknifes(n)

    stat = np.empty(shape, dtype=dtype)

    # Do jackknife resampling -- estimate statistic w/ each observation left out
    for trial,sel in enumerate(resamples):
        # Index into <axis> of data and stat, with ':' for all other axes
        slices_in   = axis_index_slices(axis, sel, ndim)
        slices_out  = axis_index_slices(axis, [trial], ndim)
        stat[slices_out] = func(data1[slices_in], data2[slices_in], *args, **kwargs)

    return stat


def jackknife_to_pseudoval(x, xjack, n):
    """
    Calculate single-trial pseudovalues from leave-one-out jackknife estimates

    Parameters
    ----------
    x : ndarray, shape=Any
        Statistic computed on full observed data. Any arbitrary shape.

    xjack : ndarray, shape=Any
        Statistic computed on jackknife resampled data

    n : int
        Number of observations (trials) used to compute x

    Returns
    -------
    pseudo : ndarray, shape=Any
        Single-trial jackknifed pseudovalues. Same shape as xjack.
    """
    return n*x - (n-1)*xjack


# =============================================================================
# Data simulation and testing functions
# =============================================================================
def simulate_multichannel_oscillation(n_chnls, *args, **kwargs):
    """
    Generate synthetic multichannel data with oscillations at given parameters.

    For each channel, generate multiple trials with constant oscillatory signal +
    random additive Gaussian noise. Parameters can be shared across channels or set
    independently for each channel.

    Parameters
    ----------
    n_chnls : int
        Number of channels to simulate

    *args :
    **kwargs :
        Rest of place and keyword arguments are passed to :func:`simulate_oscillation`
        for each channel. Each argument can be given in one of two forms:

        (1) Scalar. A single value equivalent to the same argument to simulate_oscillation().
            That same value will be copied for all `n_chnls` channels.
        (2) Array-like, shape=(n_chnls,). List with different values for each channel,
            which will be iterated through.

        See :func:`simulate_oscillation` for details on arguments.

        Exceptions: If a value is set for `seed`, it will be used to set the
        random number generator only ONCE at the start of this function, so that
        the generation of all channel signals follow a reproducible random sequence.

        Simulated data must have same shape for all channels, so all channels must
        have the same value set for `time_range`, `smp_rate`, and `n_trials`.

    Returns
    -------
    data : ndarray, shape=(n_timepts,n_trials,n_chnls)
        Simulated multichannel data
    """
    # Set a single random seed, so all multichannel data follows reproducible sequence
    seed = kwargs.pop('seed',None)
    if seed is not None:
        if not np.isscalar(seed) and (len(seed) > 1):
            seed = seed[0]
            print("Multiple values not permitted for <seed>. Using first one.")
        set_random_seed(seed)

    # Ensure all channels have same values for these parameters that determine data size
    for param in ['time_range','smp_rate','n_trials']:
        assert (param not in kwargs) or np.isscalar(kwargs[param]) or \
                np.allclose(np.diff(kwargs[param]), 0), \
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
# Other helper functions
# =============================================================================
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
