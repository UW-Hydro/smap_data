
# This script applies quality control for GPM IMERG data

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import os
import timeit

from utils import setup_output_dirs


# ========================================== #
# Parameter setting
# ========================================== #
# Original IMERG nc
gpm_nc = '/civil/hydro/ymao/smap_data/tools/prepare_GPM/output/L3_Final_global.12hour.36km/prec.20150401_20180331.nc'
# SMAP domain nc
domain_nc = '/civil/hydro/ymao/smap_data/param/domain/smap.domain.global.nc'

# Output basedir
output_basedir = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output'


# ========================================== #
# Set up directories
# ========================================== #
output_dir = setup_output_dirs(output_basedir, mkdirs=['data'])['data']


# ========================================== #
# Load original GPM data
# ========================================== #
print('Load data...')
# Load GPM
ds_gpm = xr.open_dataset(gpm_nc)
da_gpm = ds_gpm['PREC']
da_gpm.load()
da_gpm.close()
# Load domain
da_domain = xr.open_dataset(domain_nc)['mask']
# Extract dates - for output naming purpose
start_time = pd.to_datetime(da_gpm['time'].values[0])
end_time = pd.to_datetime(da_gpm['time'].values[-1])


# ========================================== #
# Exclude all-zero precipitation pixels
# ========================================== #
print('Exclude all-zero precipitation pixels...')
da_prec_max = da_gpm.max(dim='time')

prec = da_gpm.values
prec[:, da_prec_max<10e-4] = np.nan
da_gpm[:] = prec


# ========================================== #
# Save to file
# ========================================== #
# --- Save new GPM data to file --- #
print('Save to file...')
ds_gpm_new = xr.Dataset({'PREC': da_gpm})
ds_gpm_new.to_netcdf(
    os.path.join(output_dir,
                 'prec.qc.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')
