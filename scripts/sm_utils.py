
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
from sklearn.metrics import r2_score
import scipy.stats
import pickle
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def regression_time_series_wrap(run_pixel, lat_ind, lon_ind, ts_smap, ts_prec,
                                regression_type, X_version, seed, **kwargs):
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
    regression_type: <string>
        Options: linear; lasso
    X_version: <str>
    seed: <int>
        Random seed for the ts

    Returns
    ----------
    dict_results_ts: <dict>
        A dict of results (or None for an inactive pixel)
    '''

    if run_pixel == 1:
        print(lat_ind, lon_ind)
        dict_results_ts = regression_time_series(
            lat_ind, lon_ind, ts_smap, ts_prec, regression_type, X_version, seed, **kwargs)
    else:
        print(lat_ind, lon_ind, 'skip')
        dict_results_ts = None

    return dict_results_ts


def regression_time_series_chunk_wrap(lon_ind_start, lon_ind_end,
                                      da_domain_chunk, da_smap_chunk, da_prec_chunk,
                                      regression_type, X_version, seed_chunk, **kwargs):
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
    regression_type: <string>
        Options: linear; lasso
    X_version: <str>
    seed_chunk: <int>
        Random seed for the chunk

    Returns
    ----------
    dict_results_chunk: <dict>
        {latind_lonind: dict_results} for each pixel in the chunk
    '''

    # Print out longitude chunk
    print('Running for chunk lon_ind from {} to {}'.format(lon_ind_start, lon_ind_end))

    # Excecute
    results_list = \
        [regression_time_series_wrap(
            int(da_domain_chunk[lat_ind, lon_ind].values),
            lat_ind,
            lon_ind_start + lon_ind,  # this is only for printing purpose
            da_smap_chunk[:, lat_ind, lon_ind].to_series(),
            da_prec_chunk[:, lat_ind, lon_ind].to_series(),
            regression_type,
            X_version,
            seed_chunk * 10000 + (lat_ind * len(da_domain_chunk['lon']) + lon_ind),  # ensure the seeds here are different
                                                                                     #for close-by seed_chunk
            **kwargs)
        for lat_ind in range(len(da_domain_chunk['lat']))
        for lon_ind in range(len(da_domain_chunk['lon']))]
    
    # Reorganize chunk results into dict - {latind_lonind: dict_results}
    # Only save not-None pixels
    dict_results_chunk = {}
    dict_discarded_chunk = {"TooFew": {},
                            "PnotY": {},
                            "V2Corr": {}}  # {discard_reason: {lat_lon: 1}}
    for lat_ind in range(len(da_domain_chunk['lat'])):
        for lon_ind in range(len(da_domain_chunk['lon'])):
            result = results_list.pop(0)
            lon_ind_whole_domain = lon_ind_start + lon_ind
            # If None (pixel not run), skip
            if result is None:
                continue
            # If result is string, this pixel is discarded; record reason
            elif isinstance(result, str):
                dict_discarded_chunk[result]['{}_{}'.format(lat_ind, lon_ind_whole_domain)] = 1
            # Else, this pixel is fitted
            else:
                dict_results_chunk['{}_{}'.format(lat_ind, lon_ind_whole_domain)] = result

    return dict_results_chunk, dict_discarded_chunk


def regression_time_series(lat_ind, lon_ind, ts_smap, ts_prec,
                           regression_type, X_version, seed, **kwargs):
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
    regression_type: <str>
        Options: linear; lasso; ridge
    X_version: <str>
        # v1: [SM, P]
        # v2: [SM, P, SM*P]
    seed: <int>
        Random seed

    ### **kwargs ###
    alpha: <float> (only needed if regression_type = lasso or ridge)
        alpha paramter in Lasso fitting
    standardize: <bool>
        Whether to standardize X and center Y; default: True
    cross_vali: <bool>
        If true, return averaged evaluation scores from k-fold cross-validation
        If false, return evaluation scores over the entire dataset (that is also used for fitting)

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
        s_prec_step = df['prec'].iloc[smap_ind[i-1]:smap_ind[i]]
        if s_prec_step.isnull().any():  # Discard timesteps with NAN precipitation data
            continue
        # Calculate cumulative precipitation (NOTE: prec is time-beginning timestamp
        prec_sum = s_prec_step.sum()  # [mm]

        # --- Calculate Y and X elements --- #
        dt = (smap_ind[i] - smap_ind[i-1]) * 12  # delta t [hour]
        # Calculate y
        sm = df.iloc[smap_ind[i], :]['sm']
        sm_last = df.iloc[smap_ind[i-1], :]['sm']
        y = (sm - sm_last) / dt  # [mm/hour]
        Y.append(y)
        # Calculate x
        prec = prec_sum / dt  # [mm/hour]
        if X_version == 'v1' or X_version == 'v1_intercept':
            x = [sm_last, prec]
        elif X_version == 'v2' or X_version == 'v2_intercept':
            x = [sm_last, prec, sm_last * prec]
        elif X_version == 'v3':
            x = [sm_last, prec, sm_last * prec, sm_last * sm_last]
        elif X_version == 'v4':
            x = [sm_last * sm_last, prec, sm_last * prec]
        else:
            raise ValueError('Input X_version = {} unrecognizable!'.format(X_version))
        X.append(x)
        # Save time
        times.append(df.index[smap_ind[i]])
    Y = np.asarray(Y)
    X = np.asarray(X)
    times = pd.to_datetime(times)

    # --- If there are <= 100 data points, discard this cell --- #
    if len(Y) <= 100:
        print('Too few valid data points for pixel {} {} - discard!'.format(lat_ind, lon_ind))
        return "TooFew"

    # --- Quality control for precipitation data --- #
    # NOTE: assume the second column in X is precipitation!!!
    list_deleted_columns = []
    n = len(Y)
    n_coef = X.shape[1]
    X_prec = X[:, 1]
    # 1) Check if precipitation (at > 0 timesteps) is positively correlated with Y
    #    if not, drop all precipitation terms --- #
    r = np.corrcoef(X_prec[X_prec>0], Y[X_prec>0])[0, 1]
    t = r * np.sqrt((n-2) / (1-r*r))
    # Critical t (H0: r=0; H1: r>0. One-sided; alpha=0.1)
    t_critical = scipy.stats.t.ppf(0.9, df=n-2)
    if t < t_critical:  # zero r
        print('Precipitation not positively correlated with Y for pixel {} {} - discard this pixel'.format(
            lat_ind, lon_ind))
        return "PnotY"
#        if X_version == 'v1':
#            list_deleted_columns.append(1)
#        elif X_version == 'v2':
#            list_deleted_columns.append(1)
#            list_deleted_columns.append(2)
    # 2) If X_version = v2 and P and SM*P columns are highly correlated, discard this pixel for this v2 run
    if X_version == 'v2' or X_version == 'v2_intercept':
#    for i in range(n_coef-1):
#        for j in range(i+1, n_coef):
        if(np.corrcoef(X[:, 1], X[:, 2])[0, 1] >= 0.99):
            print(('X_version v2 with P and SM*P columns have correlation coefficient '
                   '>= 0.99 for pixel {} {} - drop this pixel for v2').format(
                        lat_ind, lon_ind))
            return "V2Corr"
                # Record the second column to be deleted
                #list_deleted_columns.append(j)
    # Drop columns
    if len(list_deleted_columns) > 0:
        X = np.delete(X, list_deleted_columns, axis=1)

    # --- Standardize X and center Y, if specified --- #
    if kwargs['standardize'] is True:
        # Standardize X
        X_mean = np.mean(X, axis=0)  # [n_coef]
        X = X - X_mean
        X_std = np.std(X, axis=0)  # [n_coef]
        X_std[np.all(X==0, axis=0)] = 1  # if an X column is all zero, do not standardize
        X = X / X_std
        # Center Y - in our case, Y should be already be around zero!
        Y_mean = np.mean(Y)
        Y = Y - Y_mean

    # --- Run regression --- #
    # Prepare regressor
    if regression_type == 'linear':
        if X_version == 'v1_intercept' or X_version == 'v2_intercept':
            reg = linear_model.LinearRegression(fit_intercept=True)
        else:
            reg = linear_model.LinearRegression(fit_intercept=False)
    # NOTE: the following did not incorporate intercept yet
    elif regression_type == 'lasso':
        reg = linear_model.Lasso(alpha=kwargs['alpha'],
                                 fit_intercept=False)
    elif regression_type == 'ridge':
        reg = linear_model.Ridge(alpha=kwargs['alpha'],
                                 fit_intercept=False)
    # Fit data
    model = reg.fit(X, Y)

    # --- Calculate statistics --- #
    # 1) If cross_vali is False, directly calculate stats on the entire dataset (that is also used for fitting)
    if kwargs['cross_vali'] is False:
        Y_pred = model.predict(X)
        # R2
        R2_final = r2_score(Y, Y_pred)
        # RMSE
        RMSE_final = rmse(Y, Y_pred)
    # 2) If cross_vali is True, run cross validation
    # NOTE: the returned model coefficients are still fitted on the ENTIRE dataset!!!
    # Only the evaluation stats are from cross-validataion
    else:
        n = len(Y)  # Total number of data points
        k_fold = kwargs['k_fold']
        # --- Calculate size of training and validation sets --- #
        n_vali_approx = int(n / k_fold)
        # --- Loop over each fold --- #
        list_R2_vali = []
        list_rmse_vali = []
        for k in range(k_fold):
            # Shuffle data
            ind = np.arange(n)
            rng = np.random.RandomState(seed+k)
            rng.shuffle(ind)  # Shuffle indices
            ind_vali = ind[:n_vali_approx]
            ind_train = ind[n_vali_approx:]
            # Extract training and validation data
            X_train = X[ind_train, :]
            Y_train = Y[ind_train]
            X_vali = X[ind_vali, :]
            Y_vali = Y[ind_vali]
            # Run regression on training set
            model = reg.fit(X, Y)
            # Calculate statistics on validation set
            Y_pred_vali = model.predict(X_vali)
            ### R2
            R2_vali = r2_score(Y_vali, Y_pred_vali)
            list_R2_vali.append(R2_vali)
            ### RMSE
            rmse_vali = rmse(Y_vali, Y_pred_vali)
            list_rmse_vali.append(rmse_vali)
        # --- Average stats calculatd from all folds --- #
        R2_final = np.mean(list_R2_vali)
        RMSE_final = np.mean(list_rmse_vali)

    # --- If standardize X, convert fitted coefficients back to the original X regime --- #
    if kwargs['standardize'] is True:
        # Scale back coefficients
        model.coef_ = model.coef_ / X_std
        # Calculate intercept in the original X and Y regime
        intercept = Y_mean - model.coef_ * X_mean

    # --- If dropped variable(s) in X, assign zero coef --- #
    if len(list_deleted_columns) > 0:
        indices_to_insert = np.sort(np.unique(np.asarray(
            list_deleted_columns)))
        for i in indices_to_insert:
            model.coef_ = np.insert(model.coef_, i, 0)

    # --- Put final results in dict --- #
    dict_results_ts = {}
    dict_results_ts['model'] = model
    dict_results_ts['R2'] = R2_final
    dict_results_ts['RMSE'] = RMSE_final
    if kwargs['standardize'] is True:
        dict_results_ts['intercept'] = intercept
        dict_results_ts['X_std'] = X_std
#    dict_results_ts['X'] = X
#    dict_results_ts['Y'] = Y
#    dict_results_ts['times'] = times
#    dict_results_ts['resid'] = resid
    
    return dict_results_ts


def add_gridlines(axis,
                  xlocs=[-150, -100, -50, 0,
                         50, 100, 150],
                  ylocs=[-80, -60, -40, -20, 0, 20, 40, 60, 80],
                  alpha=1):
    gl = axis.gridlines(draw_labels=True, xlocs=xlocs, ylocs=ylocs,
                        alpha=alpha)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return gl


def rmse(true, est):
    ''' Calculates RMSE of an estimated variable compared to the truth variable

    Parameters
    ----------
    true: <np.array>
        A 1-D array of time series of true values
    est: <np.array>
        A 1-D array of time series of estimated values (must be the same length of true)

    Returns
    ----------
    rmse: <float>
        Root mean square error

    Require
    ----------
    numpy
    '''

    rmse = np.sqrt(sum((est - true)**2) / len(true))
    return rmse
