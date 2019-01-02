
# This script subsample GLDAS 12H preciptation to have the same
# missing timesteps as IMERG

import sys
import subprocess
import pandas as pd
import os
import numpy as np
import xarray as xr
from sklearn.neighbors import NearestNeighbors


# ======================================================= #
# Process command line argument
# ======================================================= #
start_date = sys.argv[1]  # YYYYMMDD
end_date = sys.argv[2]  # YYYYMMDD
# 12H GPM nc - after quality control; used to subsample GLDAS PREC data
gpm_12H_nc = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/data/prec.qc_exclude_arid.20150401_20180331.nc'


# ======================================================= #
# Other parameters
# ======================================================= #
# Output dir for aggregated data
output_agg_dir = '../aggregated_variables'


# ======================================================= #
# Load in concatenated 12-hourly GLDAS prec data
# ======================================================= #
print('Load 12-hourly prec data')
# Load SM data
da_prec = xr.open_dataset(os.path.join(
    output_agg_dir,
    'prec.{}_{}.12H.nc'.format(start_date, end_date)))['PREC']
da_prec.load()
da_prec.close()


# ======================================================= #
# Load in GPM data
# ======================================================= #
print('Load GPM data')
# Load SMAP data
da_gpm = xr.open_dataset(gpm_12H_nc)['PREC']
da_gpm.load()
da_gpm.close()


# ======================================================= #
# Select IMERG timesteps from GLDAS prec and save
# NOTE: the timesteps are not exactly equivalent, since
# GPM is local solar time while GLDAS is UTC!
# ======================================================= #
# --- Construct 2D coordinate list for each GLDAS and IMERG pixel --- #
# GLDAS pixels
lonlon, latlat = np.meshgrid(da_prec['lon'], da_prec['lat'])
n_pixels_gldas = len(da_prec['lat']) * len(da_prec['lon'])
latlat_flat = latlat.reshape([n_pixels_gldas])
lonlon_flat = lonlon.reshape([n_pixels_gldas])
Y_gldas = np.stack([lonlon_flat, latlat_flat],
                   axis=1)  # [n_pixels_gldas, 2]
# GPM pixels
lonlon_gpm, latlat_gpm = np.meshgrid(da_gpm['lon'],
                                       da_gpm['lat'])
n_pixels_gpm = len(da_gpm['lat']) * len(da_gpm['lon'])
latlat_flat_gpm = latlat_gpm.reshape([n_pixels_gpm])
lonlon_flat_gpm = lonlon_gpm.reshape([n_pixels_gpm])
X_gpm = np.stack([lonlon_flat_gpm, latlat_flat_gpm],
                  axis=1)  # [n_pixels_gpm, 2]

# --- Map each coarse GLDAS pixel to the nearest finer GPM pixel --- #
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
    X_gpm)
distances, indices = nbrs.kneighbors(Y_gldas)
Y_gldas_pred = X_gpm[indices.squeeze()]  # [n_pixels_gldas, 2]
Y_gldas_pred = Y_gldas_pred.reshape(
    [len(da_prec['lat']),
     len(da_prec['lon']), 2])  # [lat_gldas, lon_gldas, 2]

# --- Assign GLDAS data new latlon coordinates (GPM coordinate) --- #
# New lats and check
lat_new = Y_gldas_pred[:, 0, 1]  # extract lats of the first lon
for i in range(len(da_prec['lon'])):
    lat_new_tmp = Y_gldas_pred[:, 0, 1]
    if np.sum(~(lat_new_tmp == lat_new)) != 0:
        print('Not all new lats are the same for each lon!')
# New lons and check
lon_new = Y_gldas_pred[0, :, 0]  # extrat lons of the first lat
for i in range(len(da_prec['lat'])):
    lon_new_tmp = Y_gldas_pred[0, :, 0]
    if np.sum(~(lon_new_tmp == lon_new)) != 0:
        print('Not all new lons are the same for each lat!')

# Subset GPM domain to GLDAS-mapped pixels
da_gpm_subset = da_gpm.loc[:, lat_new, lon_new]

# Assign new coordinates to the nearest GPM coord, and downsample temporally
da_prec_remapped = da_prec.copy(deep=True)
da_prec_remapped['lat'] = lat_new
da_prec_remapped['lon'] = lon_new
da_prec_downsampled = da_prec_remapped.where(~np.isnan(da_gpm_subset))

# Assign back the original GLDAS coords
da_prec_downsampled['lat'] = da_prec['lat']
da_prec_downsampled['lon'] = da_prec['lon']

# Save to file
ds_prec_downsampled = xr.Dataset({'PREC': da_prec_downsampled})
ds_prec_downsampled.to_netcdf(os.path.join(
    output_agg_dir,
    'prec.{}_{}.gpm_freq.exclude_arid.nc'.format(
        start_date, end_date)))

