
# This script aggregates 3H SM and prec data to 12H timestep (6AM & 6PM UTC)

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


# ======================================================= #
# Other parameters
# ======================================================= #
# Output dir for aggregated data
output_agg_dir = '../aggregated_variables'


# ======================================================= #
# Load in concatenated 3-hourly data
# ======================================================= #
print('Load 3-hourly data')
# Load SM data
da_sm = xr.open_dataset(os.path.join(output_agg_dir,
                                     'soil_moisture.{}_{}.3H.nc'.format(start_date, end_date)))['soil_moisture']
da_sm.load()
da_sm.close()
# Load precipitation data
da_prec = xr.open_dataset(os.path.join(output_agg_dir,
                                       'prec.{}_{}.3H.nc'.format(start_date, end_date)))['PREC']
da_prec.load()
da_prec.close()


# ======================================================= #
# Aggregate and save
# ======================================================= #
# --- Construct to final timestep --- #
times = pd.date_range(
    pd.to_datetime(start_date)+pd.DateOffset(hours=6),
    pd.to_datetime(end_date)+pd.DateOffset(hours=18),
    freq='12H')

# --- Extract timesteps for SM data --- #
print('Aggregate and save SM data')
# Extract
da_sm_agg = da_sm.loc[times, :, :]
# Save to file
ds_sm_agg = xr.Dataset({'soil_moisture': da_sm_agg})
ds_sm_agg.to_netcdf(os.path.join(
    output_agg_dir,
    'soil_moisture.{}_{}.12H.nc'.format(start_date, end_date)))

# --- Aggregate precipitation data to 12H (6AM & 6PM UTC) --- #
print('Aggregate and save precipitation data')
da_prec_agg = da_prec.resample(
    dim='time', freq='12H', how='sum',
    closed='left', label='left', base=6)
# Save to file
ds_prec_agg = xr.Dataset({'PREC': da_prec_agg})
ds_prec_agg.to_netcdf(os.path.join(
    output_agg_dir,
    'prec.{}_{}.12H.nc'.format(start_date, end_date)))


