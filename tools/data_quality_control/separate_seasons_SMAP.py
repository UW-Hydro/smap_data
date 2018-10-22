
# This script separates seasons of SMAP data
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
# Input SMAP nc
#smap_nc = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/data/smap.PM_rescaled.20150331_20171231.nc'  # all SMAP retrievals
smap_nc = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/data/smap.recom_qual.PM_rescaled.20150401_20180331.nc'  # all SMAP retrievals

# SMAP domain nc
domain_nc = '/civil/hydro/ymao/smap_data/param/domain/smap.domain.global.nc'

# Output basedir 
output_basedir = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/'
# Output prefix - e.g., ".YYYYMMDD_YYYYMMDD.Jun-Sep.nc" will be appended
output_prefix = 'smap.recom_qual'


# ========================================== #
# Set up directories
# ========================================== #
output_dir = setup_output_dirs(output_basedir, mkdirs=['data'])['data']

output_plot_dir = setup_output_dirs(
    output_basedir, mkdirs=['plot'])['plot']


# ========================================== #
# Load SMAP and GPM data
# ========================================== #
print('Load data...')
# Load SMAP
ds_smap = xr.open_dataset(smap_nc)
da_smap = ds_smap['soil_moisture']
da_smap.load()
da_smap.close()
# Load domain
da_domain = xr.open_dataset(domain_nc)['mask']
# Extract dates - for output naming purpose
start_time = pd.to_datetime(da_smap['time'].values[0])
end_time = pd.to_datetime(da_smap['time'].values[-1])


# ========================================== #
# Select Jun-Sep and Dec-Mar data - only for SMAP
# ========================================== #
print('Selecting seasons...')
da_smap_summer = da_smap.copy(deep=True)
da_smap_summer[:] = np.nan
da_smap_winter = da_smap.copy(deep=True)
da_smap_winter[:] = np.nan

for mon, da in da_smap.groupby('time.month'):
    if mon in [6, 7, 8, 9]:
        da_smap_summer.loc[da['time'], :, :] = da
    elif mon in [12, 1, 2, 3]:
        da_smap_winter.loc[da['time'], :, :] = da


# ========================================== #
# Plot total number of available SMAP and GPM obs for the seasons
# ========================================== #
print('Plotting...')
dict_da_smap = {'summer': da_smap_summer,
                'winter': da_smap_winter}
da_nobs_smap_summer = (~np.isnan(da_smap_summer)).sum(dim='time')
da_nobs_smap_winter = (~np.isnan(da_smap_winter)).sum(dim='time')
dict_da_nobs_smap = {'summer': da_nobs_smap_summer,
                     'winter': da_nobs_smap_winter}
dict_labels = {'summer': 'Jun-Sep',
               'winter': 'Dec-Mar'}

for season in ['summer', 'winter']:
    fig = plt.figure(figsize=(12, 5))
    # Set projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    gl = add_gridlines(ax, alpha=0)
    # Plot
    cs = dict_da_nobs_smap[season].where(da_domain==1).plot.pcolormesh(
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
            '{}.smap_nobs.{}_{}.{}.png'.format(
                output_prefix, start_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'),
                dict_labels[season])),
        format='png', bbox_inches='tight', pad_inches=0)


# ========================================== #
# Save to file
# ========================================== #
# --- Save seasonal SMAP data to file --- #
print('Save to file...')
for season in ['summer', 'winter']:
    ds_smap_seasonal = xr.Dataset(
        {'soil_moisture': dict_da_smap[season]})
    ds_smap_seasonal.to_netcdf(
        os.path.join(output_dir,
                     '{}.{}_{}.{}.nc'.format(
                        output_prefix,
                        start_time.strftime('%Y%m%d'),
                        end_time.strftime('%Y%m%d'),
                        dict_labels[season])),
        format='NETCDF4_CLASSIC')


