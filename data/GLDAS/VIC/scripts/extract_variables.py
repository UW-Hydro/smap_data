
# This script extracts and processes NLDAS (CLM) variables to top-5cm soil moisture and precipitation

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
# Paths
nc_3hourly_dir = '../3hourly_nc'

# Output dir for extracted variables
output_agg_dir = '../aggregated_variables'


# ======================================================= #
# Load in concatenated 3-hourly GLDAS data
# ======================================================= #
print('Load 3-hourly data')
ds_3hourly = xr.open_dataset(os.path.join(nc_3hourly_dir,
                             'GLDAS_VIC.{}_{}.nc'.format(start_date, end_date)))


# ======================================================= #
# Extract, process and save surface SM and precipitation variables
# ======================================================= #
# --- Extract 1st-layer SM variable --- #
print('Extract surface SM, process and save...')
# Extract
da_sm = ds_3hourly['var86'][:, 0, :, :]
# Convert mm to mm/mm (top 10-cm layer depth)
da_sm = da_sm / 100
da_sm.attrs['unit'] = 'mm/mm'
# Shift SM timestamp to be timestep-end
times = pd.to_datetime(da_sm['time'].values)
times = times + pd.DateOffset(hours=3)
da_sm['time'] = times
da_sm['time'].attrs['timestamp'] = 'timestep-end'
# Save to file
ds_sm = xr.Dataset({'soil_moisture': da_sm})
ds_sm.to_netcdf(os.path.join(output_agg_dir,
                             'soil_moisture.{}_{}.3H.nc'.format(start_date, end_date)),
                format='NETCDF4_CLASSIC')

# --- Extract precipitation --- #
print('Extract precipitation, process and save...')
# Extract snowfall and rainfall, and sum
da_snowfall = ds_3hourly['var131']
da_rainfall = ds_3hourly['var132']
da_prec = da_snowfall + da_rainfall
# Conver unit: mm/s -> mm/step
da_prec = da_prec * 3600 * 3
da_prec.attrs['unit'] = 'mm/step'
# Save to file
ds_prec = xr.Dataset({'PREC': da_prec})
ds_prec.to_netcdf(os.path.join(output_agg_dir,
                             'prec.{}_{}.3H.nc'.format(start_date, end_date)),
                  format='NETCDF4_CLASSIC')

# --- Extract SWE --- #
print('Extract SWE, process and save...')
# Extract snowfall and rainfall, and sum
da_swe = ds_3hourly['var65']
da_swe.attrs['unit'] = 'mm'
# Save to file
ds_swe = xr.Dataset({'SWE': da_swe})
ds_swe.to_netcdf(os.path.join(output_agg_dir,
                             'swe.{}_{}.3H.nc'.format(start_date, end_date)),
                 format='NETCDF4_CLASSIC')
