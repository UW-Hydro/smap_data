
import os
from collections import OrderedDict
import xarray as xr
import numpy as np
import scipy.stats
import pandas as pd


def setup_output_dirs(out_basedir, mkdirs=['results', 'state',
                                            'logs', 'plots']):
    ''' This function creates output directories.

    Parameters
    ----------
    out_basedir: <str>
        Output base directory for all output files
    mkdirs: <list>
        A list of subdirectories to make

    Require
    ----------
    os
    OrderedDict

    Returns
    ----------
    dirs: <OrderedDict>
        A dictionary of subdirectories
    '''
    dirs = OrderedDict()
    for d in mkdirs:
        dirs[d] = os.path.join(out_basedir, d)

    for dirname in dirs.values():
        os.makedirs(dirname, exist_ok=True)

    return dirs


def calculate_ecdf_percentile(value, data):
    ''' Calculates ECDF percentile using Weibull plotting position.
        "value" must be in "data"

    Parameters
    ----------
    value: <float>
        A value whose ECDF quantile is to be calculated
    data: <np.array>
        Empirical data points

    Return
    ----------
    ecdf_percentile: <float>
        Percentile from the ECDF
    '''

    data_sorted = sorted(data)
    if value not in data:
        raise ValueError("value must be in data!")
    ind, = np.where(data_sorted == value)
    # If the input data point has duplicated value in
    # the input data array, take the mean sorted index of it
    if len(ind) > 1:
        ind = np.mean(ind)
    ecdf_percentile = (int(ind) + 1) / (len(data_sorted) + 1)

    return ecdf_percentile


def construct_seasonal_window_time_indices(times):
    ''' Construct seasonal window for each timestamp in
        the intput list of times.
        (each timestamp will map to 91-day-window of all years)

    Parameters
    ----------
    times: <pd.datatimes>
        Timestamps of the data

    Returns
    ----------
    dict_window_time_indices: <dict>
        Dict of all times in a 91-day-of-all-years window
        key: input time
        value: an array of all timestamps in the window from all years
    '''

    # Extract all indices for each (month, day) of a full year
    d_fullyear = pd.date_range('20160101', '20161231')
    dayofyear_fullyear = [(d.month, d.day) for d in d_fullyear]
    list_dayofyear_index = [(times.month==d[0]) & (times.day==d[1])
                            for d in dayofyear_fullyear]
    keys = dayofyear_fullyear
    values = list_dayofyear_index
    dict_dayofyear_index = dict(zip(keys, values))

    # Calculate window mean and std values for each (month, day)
    dict_window_time_indices = {}  # {time: all times in windows}
    for d in times:
        # Identify (month, day)s in a 91-day window centered around the current day
        d_window = pd.date_range(d.date() - pd.DateOffset(days=45),
                                 d.date() + pd.DateOffset(days=45))
        dayofyear_window = [(d.month, d.day) for d in d_window]
        # Extract all time points in the window of all years
        times_window = \
            [list(times[dict_dayofyear_index[d]])
             for d in dayofyear_window]
        list_times_window = []
        for l in times_window:
            list_times_window += l
        times_window = pd.to_datetime(list_times_window).sort_values()
        # Put final 91-day-of-year window timestamps into dict
        dict_window_time_indices[d] = times_window
    
    return dict_window_time_indices


def rescale_SMAP_PM2AM_ts(ts_AM, ts_PM, dict_window_time_indices):
    ''' Rescale a ts of PM SMAP to AM using seasonal CDF matching
    
    Parameters
    ----------
    ts_AM: <pd.Series>
        SMAP AM ts
    ts_PM: <pd.Series>
        SMAP PM ts
    dict_window_time_indices: <dict>
        Dict of all times in a 91-day-of-all-years window
        key: input time
        value: an array of all timestamps in the window from all years
    
    Returns
    ----------
    ts_PM_rescaled: <pd.Series>
        Rescaled SMAP PM ts
    '''

    array_rescaled = np.asarray(
        [scipy.stats.mstats.mquantiles(
            ts_AM[dict_window_time_indices[t]].dropna(),
            calculate_ecdf_percentile(
                ts_PM[t],
                ts_PM[dict_window_time_indices[t]].dropna().values),
            alphap=0, betap=0)
         if (~np.isnan(ts_PM[t])) else np.nan
         for t in pd.to_datetime(ts_PM.index)])
    ts_PM_rescaled = pd.Series(
        array_rescaled,
        index=ts_PM.index)
    return ts_PM_rescaled


def rescale_SMAP_PM2AM_ts_wrap(run_pixel, lat_ind, lon_ind, ts_AM, ts_PM, dict_window_time_indices):
    ''' Wrapping function for parallelizing - Rescale a ts of PM SMAP to AM using seasonal CDF matching
    
    Parameters
    ----------
    run_pixel: <int>
        1 for running this pixel; 0 for not running this pixel (due to out-of-domain)
    lat_ind: <int>
        Index of lat; for printing purpose only
    lon_ind: <int>
        Index of lon; for printing purpose only
    ts_AM: <pd.Series>
        SMAP AM ts
    ts_PM: <pd.Series>
        SMAP PM ts
    dict_window_time_indices: <dict>
        Dict of all times in a 91-day-of-all-years window
        key: input time
        value: an array of all timestamps in the window from all years
    
    Returns
    ----------
    PM_rescaled: <np.array>
        Rescaled SMAP PM ts (value only)
    '''

    if run_pixel == 1:
        print(lat_ind, lon_ind)
        ts_PM_rescaled = rescale_SMAP_PM2AM_ts(ts_AM, ts_PM, dict_window_time_indices)
    else:
        print(lat_ind, lon_ind, 'skip')
        ts_PM_rescaled = ts_PM
    PM_rescaled = ts_PM_rescaled.values

    return PM_rescaled


def rescale_SMAP_PM2AM_chunk_wrap(lon_ind_start, lon_ind_end,
                                  da_domain_chunk, da_AM_chunk, da_PM_chunk,
                                  dict_window_time_indices):
    ''' Wrapping function for running a whole longitude chunk of pixels
        - Rescale a ts of PM SMAP to AM using seasonal CDF matching
    
    Parameters
    ----------
    lon_ind_start: <int>
        Index of the starting lon; for printing purpose only
    lon_ind_end: <int>
        Index of the ending lon; for printing purpose only
    da_domain_chunk: <xr.DataArray>
        Domain mask for the chunk; 1 for active pixel; 0 for inactive pixel (will skip computation)
    da_AM_chunk: <xr.DataArray>
        SMAP AM da for the chunk only; must be the same size as da_domain_chunk
    da_PM_chunk: <xr.DataArray>
        SMAP PM da for the chunk only;  must be the same size as da_domain_chunk
    dict_window_time_indices: <dict>
        Dict of all times in a 91-day-of-all-years window
        key: input time
        value: an array of all timestamps in the window from all years
    
    Returns
    ----------
    PM_rescaled_chunk: <np.array>
        Rescaled SMAP PM da for the chunk (value only)
        Dimension: [time_PM, lat, lon_chunk]
    '''

    # Print out longitude chunk
    print('Running for chunk lon_ind from {} to {}'.format(lon_ind_start, lon_ind_end))

    # Excecute
    PM_rescaled_chunk = \
        [rescale_SMAP_PM2AM_ts_wrap(
            int(da_domain_chunk[lat_ind, lon_ind].values),
            lat_ind,
            lon_ind_start + lon_ind,  # this is only for printing purpose
            da_AM_chunk[:, lat_ind, lon_ind].to_series(),
            da_PM_chunk[:, lat_ind, lon_ind].to_series(),
            dict_window_time_indices)
        for lat_ind in range(len(da_AM_chunk['lat']))
        for lon_ind in range(len(da_AM_chunk['lon']))]

    # Reshape the chunk results
    PM_rescaled_chunk = np.asarray(PM_rescaled_chunk)  # [lat*lon_chunk, time_PM]
    PM_rescaled_chunk = PM_rescaled_chunk.reshape(
        [len(da_domain_chunk['lat']),
         len(da_domain_chunk['lon']),
         -1])  # [lat, lon_chunk, time_PM]
    PM_rescaled_chunk = np.rollaxis(PM_rescaled_chunk, 2, 0)  # [time_PM, lat, lon_chunk]

    return PM_rescaled_chunk

