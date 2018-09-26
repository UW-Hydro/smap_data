
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
import multiprocessing as mp

from tonic.io import read_config, read_configobj
from sm_utils import lasso_time_series, lasso_time_series_wrap


# ========================================= #
# Process command line arguments
# ========================================= #
lasso_alpha = float(sys.argv[1])
nproc = int(sys.argv[2])


# ========================================= #
# Load data
# ========================================= #
print('Loading data...')
# Domain
da_smap_domain = xr.open_dataset('/civil/hydro/ymao/smap_data/param/domain/smap.domain.global.nc')['mask']
# SMAP
ds_smap = xr.open_dataset('/civil/hydro/ymao/smap_data/tools/prepare_SMAP/output/data/soil_moisture.20150331_20171231.nc')
# GPM
ds_prec = xr.open_dataset('/civil/hydro/ymao/smap_data/tools/prepare_GPM/output/L3_Final_global.12hour.36km/prec.20150101_20171231.nc')

ds_smap.load()
ds_smap.close()
ds_prec.load()
ds_prec.close()

# Output dir
output_dir = '/civil/hydro/ymao/smap_data/output/lasso_try'


# ========================================= #
# Lasso regression
# ========================================= #
dict_results = {}  # {latind_lonind: model/X/Y/times/resid: result}

# --- If no multiprocessing --- #
if nproc == 1:
    dict_results = {} 
    for lat_ind in range(0, 406):
        for lon_ind in range(0, 964):
            print(lat_ind, lon_ind)
            # Extract SMAP ts
            ts_smap = ds_smap['soil_moisture'][:, lat_ind, lon_ind].to_series()
            # Extract GPM ts
            ts_prec = ds_prec['PREC'][:, lat_ind, lon_ind].to_series()
            # Skip no-data pixels
            if ts_smap.isnull().all() or ts_prec.isnull().all():
                continue
            # Run Lasso
            result = lasso_time_series(
                ts_smap, ts_prec, lat_ind)
            # Skip pixels with invalid model fitting
            if result[0] is None:
                continue
            # otherwise, put results in dict
            latlon = '{}_{}'.format(lat_ind, lon_ind)
            dict_results[latlon] = {}
            dict_results[latlon]['model'] = result[0]
            dict_results[latlon]['X'] = result[1]
            dict_results[latlon]['Y'] = result[2]
            dict_results[latlon]['times'] = result[3]
            dict_results[latlon]['resid'] = result[4]

# --- If multiprocessing --- #
elif nproc > 1:
    # Set up multiprocessing
    pool = mp.Pool(processes=nproc)
    # Loop
    results = {} 
    for lat_ind in range(0, 406):
        for lon_ind in range(0, 964):
            print(lat_ind, lon_ind)
            # Extract SMAP ts
            ts_smap = ds_smap['soil_moisture'][:, lat_ind, lon_ind].to_series()
            # Extract GPM ts
            ts_prec = ds_prec['PREC'][:, lat_ind, lon_ind].to_series()
            # Skip no-data pixels
            if ts_smap.isnull().all() or ts_prec.isnull().all():
                continue
            # Run Lasso
            results['{}_{}'.format(lat_ind, lon_ind)] = pool.apply_async(
                lasso_time_series,
                (ts_smap, ts_prec, lasso_alpha))
    # Finish multiprocessing
    pool.close()
    pool.join()
    # Get results
    dict_results = {}
    for latlon, r in results.items():
        result = r.get()
        # Skip pixels with invalid model fitting
        if result[0] is None:
            continue
        # Otherwise, put results in dict
        dict_results[latlon] = {}
        dict_results[latlon]['model'] = result[0]
        dict_results[latlon]['X'] = result[1]
        dict_results[latlon]['Y'] = result[2]
        dict_results[latlon]['times'] = result[3]
        dict_results[latlon]['resid'] = result[4]
        
# --- Save results to file --- #
print('Saving results to file...')
f = open(os.path.join(output_dir, 'lasso_try.full_domain.lasso_alpha_{}.txt'.format(lasso_alpha)), 'w')
f.write('lat\tlon\tcoef1\tcoef2\tcoef3\n')
for lat_ind in range(0, 406):
    for lon_ind in range(0, 964):
        if '{}_{}'.format(lat_ind, lon_ind) in dict_results:
            model = dict_results['{}_{}'.format(lat_ind, lon_ind)]['model']
            lat = ds_smap['soil_moisture'][:, lat_ind, lon_ind]['lat'].values
            lon = ds_smap['soil_moisture'][:, lat_ind, lon_ind]['lon'].values
            f.write('{:.4f}\t{:.4f}\t{:.8f}\t{:.8f}\t{:.8f}\n'.format(
                lat, lon, model.coef_[0], model.coef_[1], model.coef_[2]))
f.close() 


# ========================================= #
# Save the result dict
# ========================================= #
print('Saving dict to file...')
with open(os.path.join(output_dir, 'lasso_try.full_domain.lasso_alpha_{}.pickle'.format(lasso_alpha)), 'wb') as f:
    pickle.dump(dict_results, f)


