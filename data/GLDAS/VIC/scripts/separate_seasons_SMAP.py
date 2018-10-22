
# This script separates seasons of SM data
# Currently, separate to two seasons: Jun-Sep; Dec-Mar

import sys
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import os
import cartopy.crs as ccrs
from joblib import Parallel, delayed, parallel_backend
import timeit

from utils import setup_output_dirs, add_gridlines


# ========================================== #
# Parameter setting
# ========================================== #
# Input SM nc
sm_nc = '/civil/hydro/ymao/smap_data/data/GLDAS/VIC/aggregated_variables/3years/soil_moisture.20150401_20180331.smap_freq.nc'  # all GLDAS SM data

# Output basedir 
output_basedir = '/civil/hydro/ymao/smap_data/data/GLDAS/VIC/aggregated_variables/3years/'
# Output prefix - e.g., ".YYYYMMDD_YYYYMMDD.Jun-Sep.nc" will be appended
output_prefix = 'soil_moisture.smap_freq'


# ========================================== #
# Set up directories
# ========================================== #
output_dir = setup_output_dirs(output_basedir, mkdirs=['data_seasonal'])['data_seasonal']

output_plot_dir = setup_output_dirs(
    output_basedir, mkdirs=['plot'])['plot']


# ========================================== #
# Load SM data
# ========================================== #
print('Load data...')
# Load SMAP
ds_sm = xr.open_dataset(sm_nc)
da_sm = ds_sm['soil_moisture']
da_sm.load()
da_sm.close()
# Extract dates - for output naming purpose
start_time = pd.to_datetime(da_sm['time'].values[0])
end_time = pd.to_datetime(da_sm['time'].values[-1])


# ========================================== #
# Select Jun-Sep and Dec-Mar data - only for SMAP
# ========================================== #
print('Selecting seasons...')
da_sm_summer = da_sm.copy(deep=True)
da_sm_summer[:] = np.nan
da_sm_winter = da_sm.copy(deep=True)
da_sm_winter[:] = np.nan

for mon, da in da_sm.groupby('time.month'):
    if mon in [6, 7, 8, 9]:
        da_sm_summer.loc[da['time'], :, :] = da
    elif mon in [12, 1, 2, 3]:
        da_sm_winter.loc[da['time'], :, :] = da


# ========================================== #
# Plot total number of available SMAP and GPM obs for the seasons
# ========================================== #
print('Plotting...')
dict_da_sm = {'summer': da_sm_summer,
                'winter': da_sm_winter}
da_nobs_sm_summer = (~np.isnan(da_sm_summer)).sum(dim='time')
da_nobs_sm_winter = (~np.isnan(da_sm_winter)).sum(dim='time')
dict_da_nobs_sm = {'summer': da_nobs_sm_summer,
                     'winter': da_nobs_sm_winter}
dict_labels = {'summer': 'Jun-Sep',
               'winter': 'Dec-Mar'}

for season in ['summer', 'winter']:
    fig = plt.figure(figsize=(12, 5))
    # Set projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    gl = add_gridlines(ax, alpha=0)
    # Plot
    cs = dict_da_nobs_sm[season].plot.pcolormesh(
        'lon', 'lat', ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap='jet',
        transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs)
    cbar.set_label(
        'Number of obs. (-)',
        fontsize=20)
    plt.title(
        'Number of SMAP obs., {}'.format(dict_labels[season]),
        fontsize=20)
    # Save fig
    fig.savefig(
        os.path.join(
            output_plot_dir,
            '{}.sm_nobs.{}_{}.{}.png'.format(
                output_prefix, start_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'),
                dict_labels[season])),
        format='png', bbox_inches='tight', pad_inches=0)


# ========================================== #
# Save to file
# ========================================== #
# --- Save seasonal SMAP data to file --- #
print('Save to file...')
for season in ['summer', 'winter']:
    ds_sm_seasonal = xr.Dataset(
        {'soil_moisture': dict_da_sm[season]})
    ds_sm_seasonal.to_netcdf(
        os.path.join(output_dir,
                     '{}.{}_{}.{}.nc'.format(
                        output_prefix,
                        start_time.strftime('%Y%m%d'),
                        end_time.strftime('%Y%m%d'),
                        dict_labels[season])),
        format='NETCDF4_CLASSIC')


