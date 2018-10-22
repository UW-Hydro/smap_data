
# This script downsamples 12H SM data (6AM & 6PM UTC) to specified temporal resolution; the original 12H timestamps are preserved

import sys
import subprocess
import pandas as pd
import os
import numpy as np
import xarray as xr


# ======================================================= #
# Process command line argument
# ======================================================= #
start_date = sys.argv[1]  # YYYYMMDD
end_date = sys.argv[2]  # YYYYMMDD
agg_freq = sys.argv[3]  # e.g., 1D; 3D


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
# Select SM timesteps and save
# ======================================================= #
# --- Construct to final timestep --- #
times = pd.date_range(
    pd.to_datetime(start_date)+pd.DateOffset(hours=6),
    pd.to_datetime(end_date)+pd.DateOffset(hours=18),
    freq=agg_freq)

# --- Only keep the selected timesteps for SM --- #
# --- and set the rest to NAN --- #
# --- NOTE: preserve the original 12H timestamps --- #
da_sm_agg = da_sm.copy(deep=True)
da_sm_agg[:] = np.nan
da_sm_agg.loc[times, :, :] = da_sm.loc[times, :, :].values

# Save to file
ds_sm_agg = xr.Dataset({'soil_moisture': da_sm_agg})
ds_sm_agg.to_netcdf(os.path.join(
    output_agg_dir,
    'soil_moisture.{}_{}.{}.nc'.format(start_date, end_date, agg_freq)))

