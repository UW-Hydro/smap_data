
# This script aggregates 30-min remapped GMP to 12-hour (SMAP temporal resolution)


import sys
import pandas as pd
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tonic.io import read_config, read_configobj
from prep_forcing_utils import (to_netcdf_forcing_file_compress, setup_output_dirs,
                                remap_con)


# ======================================================= #
# Process command line argument
# ======================================================= #
cfg = read_configobj(sys.argv[1])

# Start and end time of the 12-hour timesteps 
# The 12-hour time intervals would be the SMPA 6AM - 6PM local time
start_date = pd.to_datetime(sys.argv[2])
end_date = pd.to_datetime(sys.argv[3])

start_year = start_date.year
end_year = end_date.year


# ============================================================ #
# Setup output subdirs
# ============================================================ #
output_dir = cfg['OUTPUT']['out_dir']

output_subdir_data = setup_output_dirs(
    output_dir, mkdirs=['data'])['data']

output_12hour = cfg['OUTPUT']['out_dir_12hour']
output_subdir_plots = setup_output_dirs(output_12hour, mkdirs=['plots'])['plots']


# ======================================================= #
# Aggregating GPM 30min timestep to 12 hourly (SMAP LST 6:00 - 18:00, 18:00 - 6:00)
# NOTE: different UCT times of 30-min GPM will be extracted for the 12H-intervals local time
# ======================================================= #
# --- Construct final timestamps --- #
# LST 6am and 6pm timestamps; timestamps are time-beginnig for precipitation
times_12h = pd.date_range(
    start_date+pd.DateOffset(hours=6),
    end_date+pd.DateOffset(hours=18),
    freq='12H')

# --- Construct 2D lat and lon arrays --- #
da_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])['mask']
lonlon, latlat = np.meshgrid(da_domain['lon'].values, da_domain['lat'].values)

# --- Loop over each 12-hour LST interval --- #
for time_12h in times_12h:
    print(time_12h)  # Time-beginning timestamp
    # --- load all 30-min GPM data that falls in this 12H-LST interval for any longitude
    # NOTE: this means loading more than 12 hours of UTC-based GPM data
    # Should load 49 GPM files in total
    first_UTC = time_12h - pd.DateOffset(hours=12)  # earliest UTC time in the interval
    last_UTC = time_12h + pd.DateOffset(hours=24)  # latest UTC time in the interval
    # Load all these 30-min GPM data
    dict_ds = {}
    time = first_UTC
    while time <= last_UTC:
        # Load a 30-min GPM data
        filename = os.path.join(
            output_subdir_data,
            '{}'.format(time.year),
            '{:02d}'.format(time.month),
            '{:02d}'.format(time.day),
            '{}.{:02d}{:02d}.nc'.format(time.strftime('%Y%m%d'), time.hour, time.minute))
        ds = xr.open_dataset(filename)
        dict_ds[time] = ds
        # Move to the next 30-min time
        time = time + pd.DateOffset(minutes=30)

    # --- Aggregate all possible 12-hour GPM - aggregate 12-hour data and slice every 30 min --- #
    time = first_UTC
    dict_da_12h = {}
    while (time + pd.DateOffset(hours=12)) <= last_UTC:
        # Put the 12-hour GPM ds into list
        list_ds = []
        for hour in range(0, 12):
            for minute in [0, 30]:
                # Extract the ds from the dict, and append to list
                t = time + pd.DateOffset(seconds=hour*3600+minute*60)
                list_ds.append(dict_ds[t])
        # Aggregate to 12-hour precipitation
        ds_12h = xr.concat(list_ds, dim='time')
        da_prec = ds_12h['PREC'].mean(dim='time') * 12
        da_prec.attrs['unit'] = 'mm/step'
        da_prec = da_prec.transpose('lat', 'lon')
        # Put the 12-hour-aggregated precipitation into dict
        dict_da_12h[time] = da_prec
        # Slice to the next 12-hour interval
        time = time + pd.DateOffset(minutes=30)

    # --- Select the 12-hour-aggregated GPM corresponding to the local time for different longitudes --- #
    # Initialize the final 12-hour-aggregated precipitation (in local solar time)
    prec_12h_final = np.empty([len(da_domain['lat']), len(da_domain['lon'])])
    prec_12h_final[:] = np.nan
    # Fill in the correct UTC-based GPM data according to longitude
    count = 0
    for lon in np.arange(-180, 181, 7.5):  # every 7.5 deg longitude corresponds to 30-min timezone shift
                                           # Each longitude is assigned to the closest 30-min timezone
        # Identify the 12-hour time-beginning LST timestamp for this longitude zone
        time = time_12h - pd.DateOffset(hours=lon / 180 * 12)   # Calculate UTC time to extract
        # Fill in the data for this longitude zone
        mask = ((lonlon >= lon-3.75) & (lonlon < lon+3.75))
        prec_12h_final[mask] = dict_da_12h[time].values[mask]
    # Put the final filled data into da
    da_prec_12h_final = xr.DataArray(
        prec_12h_final,
        coords=[da_domain['lat'], da_domain['lon']],
        dims=['lat', 'lon'])

    # --- Save the 12-hour da to file --- #
    # Make ds
    da_prec_12h_final.attrs['time_zone'] = 'local solar time'
    ds_prec_12h_final = xr.Dataset({'PREC': da_prec_12h_final})
    # Prepare subdir
    # Make subdir to store processed 30min data
    out_dir_time = os.path.join(
        output_12hour,
        '{}'.format(time_12h.year),
        '{:02d}'.format(time_12h.month),
        '{:02d}'.format(time_12h.day))
    if not os.path.exists(out_dir_time):
        os.makedirs(out_dir_time)
    # Save file
    out_filename = os.path.join(
        out_dir_time,
        '{}.{:02d}{:02d}.nc'.format(time_12h.strftime('%Y%m%d'),
                                    time_12h.hour, time_12h.minute))
    ds_prec_12h_final.to_netcdf(out_filename, format='NETCDF4_CLASSIC')

