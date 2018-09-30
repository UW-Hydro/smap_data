
# This script concatenates 12-hourly remapped GMP


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

# Start and end time of the 12-hour timesteps to concatenate
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


# ======================================================= #
# Concatenate 12-hourly GPM data (SMAP LST 6:00 - 18:00, 18:00 - 6:00)
# ======================================================= #
# --- Construct timestamps to concatenate --- #
# LST 6am and 6pm timestamps; timestamps are time-beginnig for precipitation
times_12h = pd.date_range(
    start_date+pd.DateOffset(hours=6),
    end_date+pd.DateOffset(hours=18),
    freq='12H')

# --- Load all 12-hour GPM data (LST interval) --- #
list_12h_alltime = []
for time_12h in times_12h:
    print(time_12h)  # Time-beginning timestamp
    # Load data
    out_dir_time = os.path.join(
        output_12hour,
        '{}'.format(time_12h.year),
        '{:02d}'.format(time_12h.month),
        '{:02d}'.format(time_12h.day))
    out_filename = os.path.join(
        out_dir_time,
        '{}.{:02d}{:02d}.nc'.format(time_12h.strftime('%Y%m%d'),
                                    time_12h.hour, time_12h.minute))
    ds = xr.open_dataset(out_filename)
    list_12h_alltime.append(ds)

# --- Concatenate all timesteps --- #
ds_12h_alltime = xr.concat(list_12h_alltime, dim='time')
ds_12h_alltime['time'] = times_12h
ds_12h_alltime['time'].attrs['time_zone'] = 'local solar time'

# --- Save to file --- #
ds_12h_alltime.to_netcdf(
    os.path.join(output_12hour,
                 'prec.{}_{}.nc'.format(start_date.strftime('%Y%m%d'),
                                        end_date.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')



