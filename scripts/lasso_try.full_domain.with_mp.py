
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
import timeit

from tonic.io import read_config, read_configobj
from sm_utils import regression_time_series_chunk_wrap


# ========================================= #
# Process command line arguments
# ========================================= #
cfg = read_configobj(sys.argv[1])
nproc = int(sys.argv[2])


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
    **kwargs)
    for lon_ind_start in np.arange(0, nlon, lon_int_interval)) # [n_chunks: {latind_lonind: dict_results}]

# --- Merge the dict_results of all chunks together --- #
dict_results = {}
for d in results:
    dict_results.update(d)


# ========================================= #
# Save the result dict
# ========================================= #
# --- Save dict --- #
print('Saving dict to file...')
file_basename = 'results.X_{}.{}.{}.{}'.format(
    X_version,
    regression_type,
    'standardize' if standardize else 'no_standardize',
    '{}.'.format(alpha) if regression_type == 'lasso' or regression_type == 'ridge' else '')

with open(os.path.join(output_dir, '{}pickle'.format(file_basename)), 'wb') as f:
    pickle.dump(dict_results, f)

# --- Extract n_coef --- #
for latlon_ind, item in dict_results.items():
    lat_ind = int(latlon_ind.split('_')[0])
    lon_ind = int(latlon_ind.split('_')[1])
    # Extract fitted coef
    fitted_coef = item['model'].coef_
    n_coef = len(fitted_coef)
    break

# --- Save results to file --- #
print('Saving results to file...')
f = open(os.path.join(output_dir, '{}txt'.format(file_basename)), 'w')
# Write header line
f.write('lat\t\tlon\t\t')
for i in range(n_coef):
    f.write('coef{}\t'.format(i+1))
f.write('\n')
# Write results
for lat_ind in range(len(da_domain['lat'])):
    for lon_ind in range(len(da_domain['lon'])):
        if '{}_{}'.format(lat_ind, lon_ind) in dict_results.keys():
            model = dict_results['{}_{}'.format(lat_ind, lon_ind)]['model']
            lat = da_domain[lat_ind, lon_ind]['lat'].values
            lon = da_domain[lat_ind, lon_ind]['lon'].values
            f.write('{:.4f}\t{:.4f}\t'.format(lat, lon))
            for i in range(n_coef):
                f.write('{:.8f}\t'.format(model.coef_[0]))
            f.write('\n')
f.close() 

