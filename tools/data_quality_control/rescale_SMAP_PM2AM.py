
# This script:
#   1) Exclude constant-value SMAP pixels
#   2) takes care the SMAP AM & PM systematic bias
#       Specifically, remap SMAP PM data to AM regime using seasonal CDF matching

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import os
import timeit

from utils import (calculate_ecdf_percentile, construct_seasonal_window_time_indices,
                   rescale_SMAP_PM2AM_ts, setup_output_dirs)


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
end_timd = pd.to_datetime(da_smap['time'].values[-1])


# ========================================== #
# Set up directories
# ========================================== #
output_dir = setup_output_dirs(output_basedir, mkdirs=['data'])['data']


# ========================================== #
# Exclude SMAP pixels with constant values
# ========================================== #
print('Excluding SMAP pixels with constant values...')
# --- Find bad pixels --- #
da_bad_smap_pixels = (da_smap.std(dim='time')==0)
# --- Set SMAP value to all NANs for these pixels --- #
smap = da_smap.values
bad_smap_pixels = da_bad_smap_pixels.values
smap[:, bad_smap_pixels] = np.nan
da_smap[:] = smap


# ========================================== #
# Seasonal rescaling PM to AM - seasonal CDF matching
# ========================================== #
print('Seasonal rescaling PM AM...')
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
da_smap_new = da_smap.copy(deep=True)
for lat_ind in range(len(da_domain['lat'])):
    for lon_ind in range(len(da_domain['lon'])):
        print(lat_ind, lon_ind)
        # --- Skip invalid pixels --- #
        if da_domain[lat_ind, lon_ind].values == 0:
            continue
        # --- Rescale PM data --- #
        ts_AM = da_AM[:, lat_ind, lon_ind].to_series()
        ts_PM = da_PM[:, lat_ind, lon_ind].to_series()
        ts_PM_rescaled = rescale_SMAP_PM2AM_ts(
            ts_AM, ts_PM, dict_window_time_indices)
        # --- Reconstruct ts --- #
        ts = da_smap[:, lat_ind, lon_ind].to_series()
        ts[ts_PM.index] = ts_PM_rescaled
        # --- Put back in da_smap --- #
        da_smap_new[:, lat_ind, lon_ind][:] = ts

# --- Save new SMAP data to file --- #
ds_smap_new = xr.Dataset({'soil_moisture': da_smap_new})
ds_smap_new.to_netcdf(
    os.path.join(output_dir,
                 'smap.PM_rescaled.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')

