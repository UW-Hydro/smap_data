
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


def lasso_time_series_wrap(ds_smap, ds_prec, lat_ind, lon_ind, lasso_alpha):
    ''' A wrap function that extract a single pixel and perform Lasso regression

    Parameters
    ----------
    ds_smap: <xr.Dataset>
        whole SMAP dataset
    ds_prec: <xr.Dataset>
        whole IMERG dataset
    lat_ind: <int>
        lat index
    lon_ind: <int>
        lon index
    lasso_alpha: <float>
        Lasso alpha parameter

    Returns
    ----------
    '''

    # Extract SMAP ts
    ts_smap = ds_smap['soil_moisture'][:, lat_ind, lon_ind].to_series()
    # Extract GPM ts
    ts_prec = ds_prec['PREC'][:, lat_ind, lon_ind].to_series()
    # Skip no-data pixels
    if ts_smap.isnull().all() or ts_prec.isnull().all():
        return None
    # Run Lasso
    model, X, Y, times, resid = lasso_time_series(
        ts_smap, ts_prec, lasso_alpha=lasso_alpha)
    if model is None:
        return None
    else:
        return model, X, Y, times, resid


def lasso_time_series(lat_ind, lon_ind, ts_smap, ts_prec, lasso_alpha, standardize=False):
    ''' Lasso regression on a time series of SMAP and GPM
    data for one pixel
    
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
        Whether to standardize X and center Y; default: False

    Returns
    ----------
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
        # --- Otherwise, calculate Y and X elements --- #
        # Calculate y
        sm = df.iloc[smap_ind[i], :]['sm']
        sm_last = df.iloc[smap_ind[i-1], :]['sm']
        y = sm - sm_last
        Y.append(y)
        # Calculate x
        prec = df.iloc[smap_ind[i-1]:smap_ind[i], :]['prec'].sum()
        x = [sm_last, prec, sm_last * prec]
        X.append(x)
        # Save time
        times.append(df.index[smap_ind[i]])
    Y = np.asarray(Y)
    X = np.asarray(X)
    times = pd.to_datetime(times)

    # --- If there are <= 10 data points, discard this cell --- #
    if len(Y) <= 10:
        print('Too few valid data points for pixel {} {} - discard!'.format(lat_ind, lon_ind))
        return None, None, None, None, None

    np.std(X, axis=0)

    # --- Standardize X and center Y --- #
    if standardize is True:
        X = X - np.mean(X, axis=0)
        X = X / np.std(X, axis=0)
        Y = Y - np.mean(Y)

    # --- Lasso regression --- #
    # Prepare Lasso regressor
    reg = linear_model.Lasso(alpha=lasso_alpha,
                             fit_intercept=False)
    # Fit data
    model = reg.fit(X, Y)
    # --- Calculate residual --- #
    resid = Y - reg.predict(X)
    
    return model, X, Y, times, resid





