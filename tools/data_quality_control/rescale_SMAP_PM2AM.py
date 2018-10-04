
# This script:
#   1) Exclude constant-value SMAP pixels
#   2) takes care the SMAP AM & PM systematic bias
#       Specifically, remap SMAP PM data to AM regime using seasonal CDF matching

import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import os
from joblib import Parallel, delayed
import timeit

from utils import (calculate_ecdf_percentile, construct_seasonal_window_time_indices,
                   rescale_SMAP_PM2AM_ts, setup_output_dirs, rescale_SMAP_PM2AM_ts_wrap)


# ========================================== #
# Read command line arguments
# ========================================== #
nproc = int(sys.argv[1])


# ========================================== #
# Parameter setting
# ========================================== #
# Original SMAP nc
smap_nc = '/civil/hydro/ymao/smap_data/tools/prepare_SMAP/output/data/soil_moisture.20150331_20171231.nc'
# SMAP domain nc
domain_nc = '/civil/hydro/ymao/smap_data/param/domain/smap.domain.global.nc'

# Output basedir
output_basedir = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output'


# ========================================== #
# Load original SMAP data
# ========================================== #
# Load SMAP
ds_smap = xr.open_dataset(smap_nc)
ds_smap.load()
ds_smap.close()
da_smap = ds_smap['soil_moisture']
# Load domain
da_domain = xr.open_dataset(domain_nc)['mask']
# Extract dates - for output naming purpose
start_time = pd.to_datetime(da_smap['time'].values[0])
end_time = pd.to_datetime(da_smap['time'].values[-1])


# ========================================== #
# Set up directories
# ========================================== #
output_dir = setup_output_dirs(output_basedir, mkdirs=['data'])['data']


## ========================================== #
## Exclude SMAP pixels with constant values
## ========================================== #
#print('Excluding SMAP pixels with constant values...')
## --- Find bad pixels --- #
#da_bad_smap_pixels = (da_smap.std(dim='time')==0)
## --- Set SMAP value to all NANs for these pixels --- #
#smap = da_smap.values
#bad_smap_pixels = da_bad_smap_pixels.values
#smap[:, bad_smap_pixels] = np.nan
#da_smap[:] = smap


# ========================================== #
# Seasonal rescaling PM to AM - seasonal CDF matching
# ========================================== #
print('Seasonal rescaling PM to AM...')
# --- Construct seasonal windows for CDF mapping --- #
times = pd.to_datetime(da_smap['time'].values)
dict_window_time_indices = construct_seasonal_window_time_indices(times)

# --- Separate AM and PM data --- #
for hour, item in da_smap.groupby('time.hour'):
    if hour == 6:
        da_AM = item
    elif hour == 18:
        da_PM = item

# --- Rescale for each pixel --- #
#da_smap_new = da_smap.copy(deep=True)
results = Parallel(n_jobs=nproc)(delayed(rescale_SMAP_PM2AM_ts_wrap)(
    int(da_domain[lat_ind, lon_ind].values),
    lat_ind,
    lon_ind,
    da_AM[:, lat_ind, lon_ind].to_series(),
    da_PM[:, lat_ind, lon_ind].to_series(),
    dict_window_time_indices)
    for lat_ind in range(len(da_domain['lat']))
    for lon_ind in range(len(da_domain['lon'])))
#    for lat_ind in range(150, 160)
#    for lon_ind in range(330, 341))

# --- Put rescaled PM back in da_smap --- #
results = np.asarray(results)  # [lat*lon, time_PM]
results = results.reshape([len(da_domain['lat']), len(da_domain['lon']), -1])  # [lat, lon, time_PM]
results = np.rollaxis(results, 2, 0)  # [time_PM, lat, lon]
da_smap.loc[da_PM['time'], :, :] = results

# --- Save new SMAP data to file --- #
ds_smap_new = xr.Dataset({'soil_moisture': da_smap})
ds_smap_new.to_netcdf(
    os.path.join(output_dir,
                 'smap.PM_rescaled.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')

