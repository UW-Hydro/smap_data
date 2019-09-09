
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
# Exclude pixels where >= 30% of the IMERG timesteps are NAN
# ========================================== #
# Percentage of missing timesteps for each pixel
da_missing_fraction = \
    np.isnan(da_gpm).sum(dim='time') / len(da_gpm['time'])
# Set all values to NAN for pixels with >=30% missing timesteps
prec = da_gpm.values
prec[:, da_missing_fraction>=0.3] = np.nan
da_gpm[:] = prec

# --- Save intermediate data to file --- #
ds_gpm_inter = xr.Dataset({'PREC': da_gpm})
ds_gpm_inter.to_netcdf(
    os.path.join(output_dir,
                 'prec.qc_exclude_arid.inter.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')

exit()


# ========================================== #
# Exclude pixels where < 10% of the IMERG timesteps are nonzero
# (this criteria mainly excludes eastern Sahara and part of Arabian Penninsular, but not much other areas)
# ========================================== #
# Percentage of raining timesteps for each pixel
da_prec_frac = (da_gpm>0).sum(dim='time') / len(da_gpm['time'])  # [lat, lon]
# Set all values to NAN for pixels with <10% raining timesteps
prec = da_gpm.values
prec[:, da_prec_frac<0.1] = np.nan
da_gpm[:] = prec


# ========================================== #
# Save to file
# ========================================== #
# --- Save new GPM data to file --- #
print('Save to file...')
ds_gpm_new = xr.Dataset({'PREC': da_gpm})
ds_gpm_new.to_netcdf(
    os.path.join(output_dir,
                 'prec.qc_exclude_arid.{}_{}.nc'.format(
                    start_time.strftime('%Y%m%d'),
                    end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')
