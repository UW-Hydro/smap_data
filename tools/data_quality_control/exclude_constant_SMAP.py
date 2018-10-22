
# This script:
#   1) Exclude constant-value SMAP pixels

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
# Parameter setting
# ========================================== #
# Original SMAP nc
#smap_nc = '/civil/hydro/ymao/smap_data/tools/prepare_SMAP/output/data/soil_moisture.20150331_20171231.nc'
smap_nc = '/civil/hydro/ymao/smap_data/tools/prepare_SMAP/output/data/soil_moisture.20150401_20180331.nc'
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
# --- Save to file --- #
print('Save to file')
ds_smap_new = xr.Dataset({'soil_moisture': da_smap})
ds_smap_new.to_netcdf(
    os.path.join(output_dir,
                 'smap.exclude_constant.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')


