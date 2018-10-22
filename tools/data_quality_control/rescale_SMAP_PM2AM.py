
# This script:
#   1) takes care the SMAP AM & PM systematic bias
#       Specifically, remap SMAP PM data to AM regime using seasonal CDF matching

import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import os
from joblib import Parallel, delayed, parallel_backend
import timeit

from utils import (calculate_ecdf_percentile, construct_seasonal_window_time_indices,
                   rescale_SMAP_PM2AM_ts, setup_output_dirs, rescale_SMAP_PM2AM_ts_wrap,
                   rescale_SMAP_PM2AM_chunk_wrap)


# ========================================== #
# Read command line arguments
# ========================================== #
nproc = int(sys.argv[1])


# ========================================== #
# Parameter setting
# ========================================== #
# Original SMAP nc (already applied quality flag or pixel quality control)
#smap_nc = '/civil/hydro/ymao/smap_data/tools/prepare_SMAP/output/data/soil_moisture.20150331_20171231.nc'
smap_nc = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/data/smap.recom_qual.20150401_20180331.nc'
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
# Chunk the global dataset to 5-pixel-longitude chunks for multiprocessing
lon_int_interval = 5
nlon = len(da_domain['lon'])
results = Parallel(n_jobs=nproc)(delayed(rescale_SMAP_PM2AM_chunk_wrap)(
    lon_ind_start,
    (lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon,
    da_domain[:, lon_ind_start:((lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon)],
    da_AM[:, :, lon_ind_start:((lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon)],
    da_PM[:, :, lon_ind_start:((lon_ind_start + lon_int_interval) if (lon_ind_start + lon_int_interval) <= nlon else nlon)],
    dict_window_time_indices)
    for lon_ind_start in np.arange(0, nlon, lon_int_interval)) # [n_chunks: [time_PM, lat, lon_chunk]]

# --- Put the rescaled PM back in da_smap for the whole domain --- #
# Put data into da_smap
print('Put results back in da for the whole domain')
# Stack longitude chunks
smap_PM_rescaled = np.concatenate(results, axis=2)  # [time_PM, lat, lon]
# Put back into da_smap
da_smap.loc[da_PM['time'], :, :] = smap_PM_rescaled

# --- Save new SMAP data to file --- #
print('Save to file')
ds_smap_new = xr.Dataset({'soil_moisture': da_smap})
ds_smap_new.to_netcdf(
    os.path.join(output_dir,
                 'smap.recom_qual.PM_rescaled.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')

