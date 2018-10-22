
# This script downsamples GLDAS 12H SM to have the same temporal gaps as SMAP
# NOTE:
#   1) The resulting temporal gaps are not strictly the same as SMAP, since SMAP is in local solar time while GLDAS is in UTC
#   2) The output downsampled GLDAS SM is cut to the SMAP temporal range, and the 12H (UTC) timestamp is preserved

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
# 12H SMAP nc; used to subsample GLDAS SM data
smap_12H_nc = '/civil/hydro/ymao/smap_data/tools/data_quality_control/output/data/smap.recom_qual.PM_rescaled.20150401_20180331.nc'


# ======================================================= #
# Other parameters
# ======================================================= #
# Output dir for aggregated data
output_agg_dir = '../aggregated_variables'


# ======================================================= #
# Load in concatenated 12-hourly data
# ======================================================= #
print('Load 12-hourly SM data')
# Load SM data
da_sm = xr.open_dataset(os.path.join(
    output_agg_dir,
    'soil_moisture.{}_{}.12H.nc'.format(start_date, end_date)))['soil_moisture']
da_sm.load()
da_sm.close()


# ======================================================= #
# Load in SMAP data
# ======================================================= #
print('Load SMAP data')
# Load SMAP data
da_smap = xr.open_dataset(smap_12H_nc)['soil_moisture']
da_smap.load()
da_smap.close()


# ======================================================= #
# Select SMAP timesteps from GLDAS SM and save
# NOTE: the timesteps are not exactly equivalent, since
# SMAP is local solar time while GLDAS is UTC!
# ======================================================= #
# --- Construct 2D coordinate list for each GLDAS and SMAP pixel --- #
# GLDAS pixels
lonlon, latlat = np.meshgrid(da_sm['lon'], da_sm['lat'])
n_pixels_gldas = len(da_sm['lat']) * len(da_sm['lon'])
latlat_flat = latlat.reshape([n_pixels_gldas])
lonlon_flat = lonlon.reshape([n_pixels_gldas])
Y_gldas = np.stack([lonlon_flat, latlat_flat],
                   axis=1)  # [n_pixels_gldas, 2]
# SMAP pixels
lonlon_smap, latlat_smap = np.meshgrid(da_smap['lon'],
                                       da_smap['lat'])
n_pixels_smap = len(da_smap['lat']) * len(da_smap['lon'])
latlat_flat_smap = latlat_smap.reshape([n_pixels_smap])
lonlon_flat_smap = lonlon_smap.reshape([n_pixels_smap])
X_smap = np.stack([lonlon_flat_smap, latlat_flat_smap],
                  axis=1)  # [n_pixels_smap, 2]

# --- Map each coarse GLDAS pixel to the nearest finer SMAP pixel --- #
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
    X_smap)
distances, indices = nbrs.kneighbors(Y_gldas)
Y_gldas_pred = X_smap[indices.squeeze()]  # [n_pixels_gldas, 2]
Y_gldas_pred = Y_gldas_pred.reshape(
    [len(da_sm['lat']),
     len(da_sm['lon']), 2])  # [lat_gldas, lon_gldas, 2]

# --- Assign GLDAS data new latlon coordinates (SMAP coordinate) --- #
# New lats and check
lat_new = Y_gldas_pred[:, 0, 1]  # extract lats of the first lon
for i in range(len(da_sm['lon'])):
    lat_new_tmp = Y_gldas_pred[:, 0, 1]
    if np.sum(~(lat_new_tmp == lat_new)) != 0:
        print('Not all new lats are the same for each lon!')
# New lons and check
lon_new = Y_gldas_pred[0, :, 0]  # extrat lons of the first lat
for i in range(len(da_sm['lat'])):
    lon_new_tmp = Y_gldas_pred[0, :, 0]
    if np.sum(~(lon_new_tmp == lon_new)) != 0:
        print('Not all new lons are the same for each lat!')

# Subset SMAP domain to GLDAS-mapped pixels
da_smap_subset = da_smap.loc[:, lat_new, lon_new]

# Assign new coordinates to the nearest SMAP coord, and downsample temporally
da_sm_remapped = da_sm.copy(deep=True)
da_sm_remapped['lat'] = lat_new
da_sm_remapped['lon'] = lon_new
da_sm_downsampled = da_sm_remapped.where(~np.isnan(da_smap_subset))

# Assign back the original GLDAS coords
da_sm_downsampled['lat'] = da_sm['lat']
da_sm_downsampled['lon'] = da_sm['lon']

# Save to file
ds_sm_downsampled = xr.Dataset({'soil_moisture': da_sm_downsampled})
ds_sm_downsampled.to_netcdf(os.path.join(
    output_agg_dir,
    'soil_moisture.{}_{}.smap_freq.nc'.format(
        start_date, end_date)))


