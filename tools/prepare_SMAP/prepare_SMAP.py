import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import h5py
import datetime as dt
import glob
import os
from scipy.stats import rankdata
import sys

from tonic.io import read_config, read_configobj
from da_utils import (setup_output_dirs, calculate_smap_domain_from_vic_domain,
                      extract_smap_static_info, extract_smap_sm,
                      extract_smap_multiple_days, edges_from_centers, add_gridlines,
                      find_global_param_value, remap_con, rescale_SMAP_domain)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Parameter setting
# ============================================================ #
start_date = pd.to_datetime(cfg['TIME']['start_date'])
end_date = pd.to_datetime(cfg['TIME']['end_date'])

output_dir = cfg['OUTPUT']['output_dir']


# ============================================================ #
# Setup output subdirs
# ============================================================ #
output_subdir_data = setup_output_dirs(output_dir, mkdirs=['data'])['data']
output_subdir_plots = setup_output_dirs(output_dir, mkdirs=['plots'])['plots']
output_subdir_tmp = setup_output_dirs(output_dir, mkdirs=['tmp'])['tmp']


# ============================================================ #
# Load and process SMAP data
# ============================================================ #
print('Loading and processing SMAP data...')
# --- Load data --- #
print('Extracting SMAP data')
# If SMAP data is already processed before, directly load
if cfg['INPUT']['smap_exist'] is True:
    # --- Load processed SMAP data --- #
    da_smap = xr.open_dataset(cfg['INPUT']['smap_unscaled_nc'])['soil_moisture']
    # --- Extract AM and PM time points --- #
    shift_hours = int(cfg['TIME']['smap_shift_hours'])
    # AM
    am_hour = (6 + shift_hours) if (6 + shift_hours) < 24 else (6 + shift_hours - 24)
    smap_times_am_ind = np.asarray([pd.to_datetime(t).hour==am_hour
                                    for t in da_smap['time'].values])
    smap_times_am = da_smap['time'].values[smap_times_am_ind]
    # PM
    pm_hour = (18 + shift_hours) if (18 + shift_hours) < 24 else (18 + shift_hours - 24)
    smap_times_pm_ind = np.asarray([pd.to_datetime(t).hour==pm_hour
                                    for t in da_smap['time'].values])
    smap_times_pm = da_smap['time'].values[smap_times_pm_ind]

# If SMAP data not processed before, load and process
else:
    # --- Load SMAP data --- #
    da_smap, da_flag = extract_smap_multiple_days(
        os.path.join(cfg['INPUT']['smap_dir'], 'SMAP_L3_SM_P_{}_*.h5'),
        start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'),
        da_smap_domain=None)
    # --- Convert SMAP time to VIC-forcing-data time zone --- #
    # --- Shift SMAP data to the VIC-forcing-data time zone --- #
    # Shift SMAP time
    shift_hours = int(cfg['TIME']['smap_shift_hours'])
    smap_times_shifted = \
        [pd.to_datetime(t) + pd.DateOffset(seconds=3600*shift_hours)
         for t in da_smap['time'].values]
    da_smap['time'] = smap_times_shifted
    da_flag['time'] = smap_times_shifted
    # --- Exclude SMAP data points after shifting that are outside of the processing time period --- #
    da_smap = da_smap.sel(
        time=slice(start_date.strftime('%Y%m%d')+'-00',
                   end_date.strftime('%Y%m%d')+'-23'))
    da_flag = da_flag.sel(
        time=slice(start_date.strftime('%Y%m%d')+'-00',
                   end_date.strftime('%Y%m%d')+'-23'))
    # --- Get a list of SMAP AM & PM time points after shifting --- #
    # AM
    am_hour = (6 + shift_hours) if (6 + shift_hours) < 24 else (6 + shift_hours - 24)
    smap_times_am_ind = np.asarray([pd.to_datetime(t).hour==am_hour
                                    for t in da_smap['time'].values])
    smap_times_am = da_smap['time'].values[smap_times_am_ind]
    # PM
    pm_hour = (18 + shift_hours) if (18 + shift_hours) < 24 else (18 + shift_hours - 24)
    smap_times_pm_ind = np.asarray([pd.to_datetime(t).hour==pm_hour
                                    for t in da_smap['time'].values])
    smap_times_pm = da_smap['time'].values[smap_times_pm_ind]

    # --- Flip lat of SMAP da to be ascending --- #
    ds_smap = xr.Dataset({'soil_moisture': da_smap,
                          'retrieval_qual_flag': da_flag})
    lat_flipped = np.flip(ds_smap['lat'].values)
    soil_moisture_flipped = np.flip(ds_smap['soil_moisture'], axis=1)
    retrieval_qual_flag_flipped = np.flip(ds_smap['retrieval_qual_flag'], axis=1)
    ds_smap['lat'] = lat_flipped
    ds_smap['soil_moisture'] = soil_moisture_flipped
    ds_smap['retrieval_qual_flag'] = retrieval_qual_flag_flipped

    # --- Save processed SMAP data to file --- #
    ds_smap.to_netcdf(
        os.path.join(output_subdir_data,
                     'soil_moisture.{}_{}.nc'.format(
                        start_date.strftime('%Y%m%d'),
                        end_date.strftime('%Y%m%d'))))
