
import sys
import pandas as pd
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xesmf as xe
import numpy as np
from sklearn import linear_model
import pickle


def lasso_time_series_wrap(run_pixel, lat_ind, lon_ind, ts_smap, ts_prec, lasso_alpha, standardize=True):
    ''' A wrap function that extract a single pixel and perform Lasso regression

    Parameters
    ----------
    run_pixel: <int>
        1 for running this pixel; 0 for not running this pixel (due to out-of-domain)
    lat_ind: <ind>
        zero-starting lat index - for printing purpose only
    lon_ind: <ind>
        zero-starting lon index - for printing purpose only
    ts_smap: <pd.Series>
        SMAP time series (with NAN)
    ts_prec: <pd.Series>
        IMERG precipitation time series
    lasso_alpha: <float>
        alpha paramter in Lasso fitting
    standardize: <bool>
        Whether to standardize X and center Y; default: True

    Returns
    ----------
    dict_results_ts: <dict>
        A dict of results (or None for an inactive pixel)
    '''

    if run_pixel == 1:
        print(lat_ind, lon_ind)
        dict_results_ts = lasso_time_series(
            lat_ind, lon_ind, ts_smap, ts_prec, lasso_alpha, standardize)
    else:
        print(lat_ind, lon_ind, 'skip')
        dict_results_ts = None

    return dict_results_ts


def lasso_time_series_chunk_wrap(lon_ind_start, lon_ind_end,
                                 da_domain_chunk, da_smap_chunk, da_prec_chunk,
                                 lasso_alpha, standardize=True):
    ''' Wrapping function for running a whole longitude chunk of pixels
        - Lasso regression of SMAP and GPM data for each pixel
    
    Parameters
    ----------
    lon_ind_start: <int>
        Index of the starting lon; for printing purpose only
    lon_ind_end: <int>
        Index of the ending lon; for printing purpose only
    da_domain_chunk: <xr.DataArray>
        Domain mask for the chunk; 1 for active pixel; 0 for inactive pixel (will skip computation)
    da_smap_chunk: <xr.DataArray>
        SMAP data (with NAN) for the chunk
    da_prec_chunk: <xr.DataArray>
        IMERG precipitation data for the chunk
    lasso_alpha: <float>
        alpha paramter in Lasso fitting
    standardize: <bool>
        Whether to standardize X and center Y; default: True

    Returns
    ----------
    dict_results_chunk: <dict>
        {latind_lonind: dict_results} for each pixel in the chunk
    '''

    # Print out longitude chunk
    print('Running for chunk lon_ind from {} to {}'.format(lon_ind_start, lon_ind_end))

    # Excecute
    results_list = \
        [lasso_time_series_wrap(
            int(da_domain_chunk[lat_ind, lon_ind].values),
            lat_ind,
            lon_ind_start + lon_ind,  # this is only for printing purpose
            da_smap_chunk[:, lat_ind, lon_ind].to_series(),
            da_prec_chunk[:, lat_ind, lon_ind].to_series(),
            lasso_alpha,
            standardize)
        for lat_ind in range(len(da_domain_chunk['lat']))
        for lon_ind in range(len(da_domain_chunk['lon']))]
    
    # Reorganize chunk results into dict - {latind_lonind: dict_results}
    # Only save not-None pixels
    dict_results_chunk = {}
    for lat_ind in range(len(da_domain_chunk['lat'])):
        for lon_ind in range(len(da_domain_chunk['lon'])):
            result = results_list.pop(0)
            if result is not None:
                lon_ind_whole_domain = lon_ind_start + lon_ind
                dict_results_chunk['{}_{}'.format(lat_ind, lon_ind_whole_domain)] = result

    return dict_results_chunk


def lasso_time_series(lat_ind, lon_ind, ts_smap, ts_prec, lasso_alpha, standardize=True):
    ''' Lasso regression on a time series of SMAP and GPM data for one pixel
    
    Parameters
    ----------
    lat_ind: <ind>
        zero-starting lat index - for printing purpose only
    lon_ind: <ind>
        zero-starting lon index - for printing purpose only
    ts_smap: <pd.Series>
        SMAP time series (with NAN)
    ts_prec: <pd.Series>
        IMERG precipitation time series
    lasso_alpha: <float>
        alpha paramter in Lasso fitting
    standardize: <bool>
        Whether to standardize X and center Y; default: True

    Returns
    ----------
    dict_results_ts: <dict>
        A dict of regression results (or None for invalid pixels). Specifically, each element is:
            model: <sklearn.linear_model.coordinate_descent.Lasso>
                fitted Lasso model
            X: <np.array>
                X matrix directly used for Lasso fitting
            Y: <np.array>
                Y vector directly used for Lasso fitting
            times: <pd.datetime>
                timestamps corresponding to the timesteps of X and Y
            resid: <np.array>
                Residual time series from the fitting
    '''
    
    # --- Put SMAP and prec into a df --- #
    ts_prec = ts_prec.truncate(before=ts_smap.index[0],
                               after=ts_smap.index[-1])
    ts_smap = ts_smap.truncate(before=ts_prec.index[0],
                               after=ts_prec.index[-1])
    df = pd.concat([ts_smap, ts_prec], axis=1,
                   keys=['sm', 'prec'])
    
    # --- Construct X and Y for regression --- #
    # X: SM(k-1), sum(P), sum(P)*SM(k-1)
    # Y: SM(k) - SM(k-1)

    # Find all time indices with SMAP observations
    smap_ind = np.argwhere(~np.isnan(df['sm']))[:, 0]
    # Construct X and Y matrices
    X = []
    Y = []
    times = []
    # Start from the second SMAP obs
    for i in range(1, len(smap_ind)):
        # --- If there the SMAP data gap is too big, skip calculation --- #
        # (Right now set the limit to 5 days)
        if smap_ind[i] - smap_ind[i-1] > 10:
            continue
        # --- Discard this timestep if precipitation data contains NAN
        # Calculate cumulative precipitation (NOTE: prec is time-beginning timestamp
        prec_sum = df.iloc[smap_ind[i-1]:smap_ind[i], :]['prec'].sum()  # [mm]
        if np.isnan(prec_sum) is True:  # Discard timesteps with NAN precipitation data
            continue

        # --- Calculate Y and X elements --- #
        dt = (smap_ind[i] - smap_ind[i-1]) * 12  # delta t [hour]
        # Calculate y
        sm = df.iloc[smap_ind[i], :]['sm']
        sm_last = df.iloc[smap_ind[i-1], :]['sm']
        y = (sm - sm_last) / dt  # [mm/hour]
        Y.append(y)
        # Calculate x
        prec = prec_sum / dt  # [mm/hour]
#        x = [sm_last, prec, sm_last * prec]
        x = [sm_last, prec]
        X.append(x)
        # Save time
        times.append(df.index[smap_ind[i]])
    Y = np.asarray(Y)
    X = np.asarray(X)
    times = pd.to_datetime(times)

    # --- If there are <= 10 data points, discard this cell --- #
    if len(Y) <= 10:
        print('Too few valid data points for pixel {} {} - discard!'.format(lat_ind, lon_ind))
        return None

    # --- Standardize X and center Y --- #
    if standardize is True:
        # Standardize X
        X = X - np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[np.all(X==0, axis=0)] = 1  # if an X column is all zero, do not standardize
        X = X / X_std
        # Center Y
        Y = Y - np.mean(Y)

    # --- Lasso regression --- #
    # Prepare Lasso regressor
    reg = linear_model.Lasso(alpha=lasso_alpha,
                             fit_intercept=False)
    # Fit data
    model = reg.fit(X, Y)
    # --- Calculate residual --- #
    resid = Y - reg.predict(X)

    # --- Put final results in dict --- #
    dict_results_ts = {}
    dict_results_ts['model'] = model
#    dict_results_ts['X'] = X
#    dict_results_ts['Y'] = Y
#    dict_results_ts['times'] = times
#    dict_results_ts['resid'] = resid
    
    return dict_results_ts





