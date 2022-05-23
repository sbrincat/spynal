""" Unit tests for plots.py module """
import pytest
import numpy as np

from spynal.tests.data_fixtures import oscillation, MISSING_ARG_ERRS
from spynal.plots import plot_line_with_error_fill, plot_lineseries, plot_heatmap

def test_plot_line_with_error_fill(oscillation):
    """ Unit tests for plot_lineseries function """
    data = oscillation.T
    data_orig = data.copy()
    n_chnls, n_timepts = data.shape

    timepts = np.arange(n_timepts)/1000
    errs = 0.1*np.ptp(data)*np.ones_like(data)
    errs_orig = errs.copy()

    # Basic test that plotted data == input data
    lines, _, _ = plot_line_with_error_fill(timepts, data)
    assert np.array_equal(data, data_orig) # Ensure input data isn't altered by function
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), data[ch,:])

    # Test with 1-sided error inputs
    lines, _, _ = plot_line_with_error_fill(timepts, data, err=errs)
    assert np.array_equal(data, data_orig) # Ensure input data isn't altered by function
    assert np.array_equal(errs, errs_orig)
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), data[ch,:])

    # Test with 2-sided (upper; lower) error inputs
    errs2 = np.empty((n_chnls*2,n_timepts))
    errs2[0::2,:] = data + errs
    errs2[1::2,:] = data - errs
    errs2_orig = errs2.copy()
    lines, _, _ = plot_line_with_error_fill(timepts, data, err=errs2)
    assert np.array_equal(data, data_orig) # Ensure input data isn't altered by function
    assert np.array_equal(errs2, errs2_orig)
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), data[ch,:])

    # Test for consistent output with specifying each line color
    lines, _, _ = plot_line_with_error_fill(timepts, data, err=errs, color=['C1','C2','C3','C4'])
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), data[ch,:])

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        lines, _, _ = plot_line_with_error_fill(timepts, data, foo=None)


def test_plot_lineseries(oscillation):
    """ Unit tests for plot_lineseries function """
    data = oscillation.T
    data_orig = data.copy()
    n_chnls, n_timepts = data.shape

    timepts = np.arange(n_timepts)/1000
    channels = np.arange(n_chnls)

    max_val = np.abs(data).max()
    scaled_data = 1.5 * data / max_val
    
    # Basic test that plotted data == input data
    lines, _ = plot_lineseries(timepts, channels, data)
    assert np.array_equal(data, data_orig) # Ensure input data isn't altered by function
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), scaled_data[ch,:] + n_chnls - (ch+1))

    # Test for consistent output with specifying each line color
    lines, _ = plot_lineseries(timepts, channels, data, color=['C'+str(j) for j in range(n_chnls)])
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), scaled_data[ch,:] + n_chnls - (ch+1))

    # Test for consistent output with inverted y-axis
    lines, _ = plot_lineseries(timepts, channels, data, origin='lower')
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), scaled_data[ch,:] + ch)

    # Test for consistent output with change in scale
    scaled_data = 0.5 * data / max_val    
    lines, _ = plot_lineseries(timepts, channels, data, scale=0.5)
    for ch in range(n_chnls):
        assert np.allclose(lines[ch][0].get_xdata(), timepts)
        assert np.allclose(lines[ch][0].get_ydata(), scaled_data[ch,:] + n_chnls - (ch+1))

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        lines, _ = plot_lineseries(timepts, channels, data, foo=None)


def test_plot_heatmap(oscillation):
    """ Unit tests for plot_heatmap function """
    data = oscillation.T
    data_orig = data.copy()
    n_chnls, n_timepts = data.shape

    timepts = np.arange(n_timepts)/1000
    channels = np.arange(n_chnls)

    # Basic test that plotted data == input data
    img, _ = plot_heatmap(timepts, channels, data)
    assert np.array_equal(data, data_orig) # Ensure input data isn't altered by function
    assert np.allclose(img.get_array().data, data)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        img, _ = plot_heatmap(timepts, channels, data, foo=None)


def test_imports():
    """ Test different import methods for plots module """
    # Import entire package
    import spynal
    spynal.plots.full_figure
    # Import module
    import spynal.plots as plots
    plots.full_figure
    # Import specific function from module
    from spynal.plots import full_figure
    full_figure
