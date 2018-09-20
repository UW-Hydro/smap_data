
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
# (time-beginning timestamp, e.g., 20150101-00 20171231-12)
start_time = pd.to_datetime(sys.argv[2])
end_time = pd.to_datetime(sys.argv[3])

start_year = start_time.year
end_year = end_time.year


# ======================================================= #
# Process time index
# ======================================================= #
#start_time = pd.to_datetime(cfg['TIME']['start_time'])
#end_time = pd.to_datetime(cfg['TIME']['end_time'])
#start_year = start_time.year
#end_year = end_time.year


# ============================================================ #
# Setup output subdirs
# ============================================================ #
output_dir = cfg['OUTPUT']['out_dir']

output_subdir_plots = setup_output_dirs(output_dir, mkdirs=['plots'])['plots']
output_subdir_data = setup_output_dirs(
    output_dir, mkdirs=['data'])['data']
output_subdir_tmp = setup_output_dirs(output_dir, mkdirs=['tmp'])['tmp']

output_12hour = cfg['OUTPUT']['out_dir_12hour']


# ======================================================= #
# Aggregating GPM 30min timestep to 12 hourly (UTC 00:00 - 12:00, 12:00 - 0:00)
# ======================================================= #
times_12h = pd.date_range(start_time, end_time, freq='12H')
list_12h_alltime = []
# --- Loop over each 12-hour timestep --- #
for time_12h in times_12h:
    print(time_12h)
    # --- Loop over each 30-min in this 12-hour timestep and load GPM data --- #
    list_ds = []
    list_times = []
    for hour in range(0, 12):
        for minute in [0, 30]:
            time = time_12h + pd.DateOffset(seconds=3600*hour+60*minute)
            list_times.append(time)
            # Load a 30-min GPM data
            filename = os.path.join(
                output_subdir_data,
                '{}'.format(time.year),
                '{:02d}'.format(time.month),
                '{:02d}'.format(time.day),
                '{}.{:02d}{:02d}.nc'.format(time.strftime('%Y%m%d'), time.hour, time.minute))
            ds = xr.open_dataset(filename)
            list_ds.append(ds)
    # Concatenate all timesteps in this 12-hour step
    ds_12h = xr.concat(list_ds, dim='time')
    # Aggregate precipitation - sum; [mm/hour] -> [mm/step]
    da_prec = ds_12h['PREC'].mean(dim='time') * 12
    da_prec.attrs['unit'] = 'mm/step'
    da_prec = da_prec.transpose('lat', 'lon')
    # Append data to final list
    list_12h_alltime.append(da_prec)

# --- Concat all 12-hour timesteps --- #
da_prec_12h_alltime = xr.concat(list_12h_alltime, dim='time')
da_prec_12h_alltime['time'] = times_12h

# --- Save to file --- #
ds_prec_12h_alltime = xr.Dataset({'PREC': da_prec_12h_alltime})
out_filename = os.path.join(output_12hour,
                            'prec.{}_{}.nc'.format(start_time.strftime('%Y%m%d'),
                                                   end_time.strftime('%Y%m%d')))
#to_netcdf_forcing_file_compress(ds_prec_12h_alltime, out_filename)
ds_prec_12h_alltime.to_netcdf(out_filename, format='NETCDF4_CLASSIC')


