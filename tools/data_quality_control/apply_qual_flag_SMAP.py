
# This script:
#   1) Exclude SMAP pixels with unrecommended retrieval quality flag (according to SMAP internal flag)

import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import os
from joblib import Parallel, delayed, parallel_backend
import timeit

from utils import setup_output_dirs


# ========================================== #
# Parameter setting
# ========================================== #
# Input SMAP nc
#smap_nc = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/data/smap.PM_rescaled.20150331_20171231.nc'
smap_nc = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/data/smap.exclude_constant.20150401_20180331.nc'
# SMAP quality control flag nc
smap_qual_flag_nc = '/civil/hydro/ymao/smap_data/tools/prepare_SMAP/output/data/soil_moisture.20150401_20180331.nc'
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
# Load SMAP quality control flag
ds_smap_qual_flag = xr.open_dataset(smap_qual_flag_nc)
ds_smap_qual_flag.load()
ds_smap_qual_flag.close()
da_smap_qual_flag = ds_smap_qual_flag['retrieval_qual_flag']
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
# Apply retrieval flag
# ========================================== #
print('Applying retrieval falg...')
da_smap_recom_qual = da_smap.where(da_smap_qual_flag==0)


# ========================================== #
# Save to file
# ========================================== #
# --- Save new SMAP data to file --- #
print('Save to file')
ds_smap_new = xr.Dataset({'soil_moisture': da_smap_recom_qual})
ds_smap_new.to_netcdf(
    os.path.join(output_dir,
                 'smap.recom_qual.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')

