
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
from joblib import Parallel, delayed
import subprocess
import timeit

from tonic.io import read_config, read_configobj
from sm_utils import regression_time_series_chunk_wrap


# ========================================= #
# Process command line arguments
# ========================================= #
cfg = read_configobj(sys.argv[1])
nproc = int(sys.argv[2])


# ========================================= #
# Set random generation seed
# ========================================= #
if 'RANDOM' in cfg:
    np.random.seed(cfg['RANDOM']['seed'])
else:
    raise ValueError('Input a random seed in the cfg file (even if no randomness is in the analysis)')


# ========================================= #
# Load data
# ========================================= #
print('Loading data...')
# Domain
da_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])['mask']
# SMAP (instantaneous time points)
da_smap = xr.open_dataset(cfg['INPUT']['smap_nc'])['soil_moisture']
# GPM (time-beginning timestamp)
da_prec = xr.open_dataset(cfg['INPUT']['gpm_nc'])['PREC']

da_smap.load()
da_smap.close()
da_prec.load()
da_prec.close()

# Output dir
output_dir = cfg['OUTPUT']['output_dir']


# ========================================= #
# Make output subdir
# ========================================= #
# Output data dir
output_data_dir = os.path.join(output_dir, 'data')
subprocess.call("mkdir -p {}".format(output_data_dir), shell=True)


# ========================================= #
# Run regression
# ========================================= #
dict_results = {}  # {latind_lonind: model/X/Y/times/resid: result}

# --- Construct kwargs --- #
regression_type = cfg['REGRESSION']['regression_type']
standardize = cfg['REGRESSION']['standardize']
X_version = cfg['REGRESSION']['X_version']
if regression_type == 'linear':
    kwargs = {'standardize': standardize}
elif regression_type == 'lasso' or regression_type == 'ridge':
    alpha = cfg['REGRESSION']['alpha']
    kwargs = {'alpha': alpha,
              'standardize': standardize}
else:
    raise ValueError('Input regression_type = {} not recognizable!'.format(regression_type))
# Cross validation
if 'cross_vali' in cfg['REGRESSION'] and cfg['REGRESSION']['cross_vali'] is True:
    cross_vali = True
    kwargs['k_fold'] = cfg['REGRESSION']['k_fold']
else:
    cross_vali = False
kwargs['cross_vali'] = cross_vali
seed_chunk_base = np.random.randint(low=10000)

# --- Chunk the global dataset to 5-pixel-longitude chunks for multiprocessing --- #
lon_int_interval = 5
nlon = len(da_domain['lon'])
results = Parallel(n_jobs=nproc)(delayed(regression_time_series_chunk_wrap)(
    lon_ind_start,
    (lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon,
    da_domain[:, lon_ind_start:((lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon)],
    da_smap[:, :, lon_ind_start:((lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon)],
    da_prec[:, :, lon_ind_start:((lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon)],
    cfg['REGRESSION']['regression_type'],
    X_version,
    seed_chunk_base + int(lon_ind_start / lon_int_interval),
    **kwargs)
    for lon_ind_start in np.arange(0, nlon, lon_int_interval)) # [n_chunks: {latind_lonind: dict_results}]

# --- Merge the dict_results of all chunks together --- #
dict_results = {}
dict_discards = {"TooFew": {},
                 "PnotY": {},
                 "V2Corr": {}}
for d_result, d_discard in results:
    dict_results.update(d_result)
    for discard in ["TooFew", "PnotY", "V2Corr"]:
        dict_discards[discard].update(d_discard[discard])


# ========================================= #
# Save the result dict
# ========================================= #
# --- Save dict --- #
print('Saving dict to file...')
file_basename = 'X_{}.{}.{}.{}'.format(
    X_version,
    regression_type,
    'standardize' if standardize else 'no_standardize',
    '{}.'.format(alpha) if regression_type == 'lasso' or regression_type == 'ridge' else '')

with open(os.path.join(output_data_dir, '{}pickle'.format(file_basename)), 'wb') as f:
    pickle.dump(dict_results, f)

# --- Save discard dict --- #
with open(os.path.join(output_data_dir, '{}discard.pickle'.format(file_basename)), 'wb') as f:
    pickle.dump(dict_discards, f)

# --- Extract n_coef --- #
for latlon_ind, item in dict_results.items():
    lat_ind = int(latlon_ind.split('_')[0])
    lon_ind = int(latlon_ind.split('_')[1])
    # Extract fitted coef
    fitted_coef = item['model'].coef_
    n_coef = len(fitted_coef)
    break


# ========================================= #
# Re-organize results and save to netCDF
# ========================================= #
# --- Prepare empty result spatial da --- #
list_coef = []  # A list of results da for each Lasso param
# Prepare an initial da to store results
init = da_domain.values
init = init.astype(float)
init[:] = np.nan
da = da_domain.copy(deep=True)
da = da.astype(float)
da[:] = init
# Fitted coefficients
for i in range(n_coef):
    da_init = da.copy(True)
    list_coef.append(da_init)
# Fitted intercept, if applicable
if X_version == 'v1_intercept' or X_version == 'v2_intercept':
    da_intercept = da.copy(True)
# Discarded
dict_da_discard = {}
for discard in ["TooFew", "PnotY", "V2Corr"]:
    dict_da_discard[discard] = da.copy(True)
# R^2
da_R2 = da.copy(True)
# RMSE
da_RMSE = da.copy(True)

# --- Extract results --- #
for latlon_ind, item in dict_results.items():
    lat_ind = int(latlon_ind.split('_')[0])
    lon_ind = int(latlon_ind.split('_')[1])
    # Extract fitted coef
    fitted_coef = item['model'].coef_
    if X_version == 'v1_intercept' or X_version == 'v2_intercept':
        fitted_intercept = item['model'].intercept_
    # Extract R^2
    R2 = item['R2']
    # Extract RMSE
    RMSE = item['RMSE']
    # Put results into da
    for i in range(n_coef):
        list_coef[i][lat_ind, lon_ind] = fitted_coef[i]
    if X_version == 'v1_intercept' or X_version == 'v2_intercept':
        da_intercept[lat_ind, lon_ind] = fitted_intercept
    da_R2[lat_ind, lon_ind] = R2
    da_RMSE[lat_ind, lon_ind] = RMSE

# --- Extract discarded pixels --- #
for discard in ["TooFew", "PnotY", "V2Corr"]:
    for latlon_ind, item in dict_discards[discard].items():
        lat_ind = int(latlon_ind.split('_')[0])
        lon_ind = int(latlon_ind.split('_')[1])
        dict_da_discard[discard][lat_ind, lon_ind] = 1

# --- Save to netCDF --- #
# coefficients
for i in range(n_coef):
    ds = xr.Dataset({'coef': list_coef[i]})
    ds.to_netcdf(os.path.join(output_data_dir,
                              '{}fitted_coef.{}.nc'.format(file_basename, i+1)),
                 format='NETCDF4_CLASSIC')
if X_version == 'v1_intercept' or X_version == 'v2_intercept':
    ds = xr.Dataset({'intercept': da_intercept})
    ds.to_netcdf(os.path.join(output_data_dir,
                              '{}fitted_intercept.nc'.format(file_basename)),
                 format='NETCDF4_CLASSIC')
# R2
ds_R2 = xr.Dataset({'R2': da_R2})
ds_R2.to_netcdf(os.path.join(output_data_dir,
                            '{}R2.nc'.format(file_basename)),
                format='NETCDF4_CLASSIC')

# RMSE
ds_RMSE = xr.Dataset({'RMSE': da_RMSE})
ds_RMSE.to_netcdf(os.path.join(output_data_dir,
                               '{}RMSE.nc'.format(file_basename)),
                  format='NETCDF4_CLASSIC')
# Discarded
for discard in ["TooFew", "PnotY", "V2Corr"]:
    ds = xr.Dataset({discard: dict_da_discard[discard]})
    ds.to_netcdf(os.path.join(output_data_dir,
                              '{}discarded_{}.nc'.format(file_basename, discard)),
                 format='NETCDF4_CLASSIC')

