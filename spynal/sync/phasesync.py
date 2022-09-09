# -*- coding: utf-8 -*-
""" Phase-based oscillatory neural synchrony analysis """
from warnings import warn
import numpy as np

from spynal.utils import axis_index_slices, setup_sliding_windows
from spynal.randstats.sampling import jackknifes
from spynal.randstats.utils import jackknife_to_pseudoval
from spynal.sync.helpers import _sync_raw_to_spectral, _sfc_raw_to_spectral


# =============================================================================
# Field-Field Synchrony functions
# =============================================================================
def plv(data1, data2, axis=0, return_phase=False, transform=None, single_trial=None,
        spec_method='wavelet', data_type=None, smp_rate=None,
        time_axis=None, taper_axis=None, keepdims=True, **kwargs):
    """
    Compute phase locking value (PLV) between raw or spectral (time-frequency) LFP data.

    PLV is a measure of phase synchrony that ignores signals amplitudes.

    PLV is the mean resultant length (magnitude of the vector mean) of phase
    differences dphi btwn phases of data1 and data2::

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
    and can be expressed (and computed efficiently) in terms of PLV and n::

        PPC = (n*PLV^2 - 1) / (n-1)

    Only parameters differing from :func:`.synchrony` are described here.

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


# =============================================================================
# Spike-Field Synchrony functions
# =============================================================================
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

    Only parameters differing from :func:`.spike_field_coupling` are described here.

    Parameters
    ----------
    width : scalar, default: 0.5 (500 ms)
        Width (in s) of sliding time windows for computing PLV

    spacing : scalar, default: set = `width` (ie exactly non-overlapping windows)
        Spacing (in s) of sliding time windows for computing PLV

    lims : array-like, shape=(2,), default: (timepts[0],timepts[-1]) (full sampled time of data)
        [Start,end] time limits (in s) for full series of sliding windows

    timewins : ndarray, shape=(n_timewins,2), default: setup_sliding_windows(width,lims,spacing)
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
    # Move time and observations/trials axes to end of arrays -> (...,n_timepts,n_obs)
    if not ((time_axis == data_ndim-2) and (axis == data_ndim-1)):
        lfpdata = np.moveaxis(lfpdata,time_axis,-1)
        lfpdata = np.moveaxis(lfpdata,axis,-1)
        spkdata = np.moveaxis(spkdata,time_axis,-1)
        spkdata = np.moveaxis(spkdata,axis,-1)
    data_shape = lfpdata.shape   # Cache data shape after axes shift
    n_timepts,n_obs = data_shape[-2], data_shape[-1] # n_obs = n_trials[*n_tapers]

    # Unwrap all other axes (incl. frequency) -> (n_data_series,n_timepts,n_obs)
    if data_ndim > 2:
        lfpdata = np.reshape(lfpdata, (-1,n_timepts,n_obs))
        spkdata = np.reshape(spkdata, (-1,n_timepts,n_obs))

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
    # Move time and trials/observations axes to original locations
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
    
    if not keepdims:
        vector_mean = vector_mean.squeeze(axis=axis)
        
    # Insert singleton axis corresponding to former location of tapers, for dimensional consistency    
    elif spec_method == 'multitaper':
        slicer = [slice(None)]*vector_mean.ndim
        slicer = slicer[:time_axis] + [np.newaxis] + slicer[time_axis:]
        vector_mean = vector_mean[tuple(slicer)]

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

    Only parameters differing from :func:`.spike_field_coupling` are described here.

    Parameters
    ----------
    width : scalar, default: 0.5 (500 ms)
        Width (in s) of sliding time windows for computing PPC

    spacing : scalar, default: set = `width` (ie exactly non-overlapping windows)
        Spacing (in s) of sliding time windows for computing PPC

    lims : array-like, shape=(2,), default: (timepts[0],timepts[-1]) (full sampled time of data)
        [Start,end] time limits (in s) for full series of sliding windows

    timewins : ndarray, shape=(n_timewins,2), default: setup_sliding_windows(width,lims,spacing)
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
# Helper functions
# =============================================================================
def plv_to_ppc(PLV, n):
    """
    Convert PLV to PPC as::
    
        PPC = (n*PLV^2 - 1)/(n-1)
    """
    return (n*PLV**2 - 1) / (n - 1)
