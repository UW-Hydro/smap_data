
import sys
import subprocess
import pandas as pd
import os
import numpy as np
import xarray as xr


# ======================================================= #
# Process command line argument
# ======================================================= #
start_date = sys.argv[1]  # YYYYMM
end_date = sys.argv[2]  # YYYYMM

start_year = int(start_date[:4])
end_year = int(end_date[:4])


# ======================================================= #
# Other parameters
# ======================================================= #
# Paths
nc_monthly_dir = '../monthly_nc'

# Output dir for extracted variables
output_agg_dir = '../aggregated_variables'


# ======================================================= #
# Load in concatenated monthly GLDAS data
# ======================================================= #
print('Load monthly data')
ds_monthly = xr.open_dataset(os.path.join(nc_monthly_dir,
                             'GLDAS_VIC.{}_{}.nc'.format(start_date, end_date)))


# ======================================================= #
# Extract, process and save mean annual variables
# ======================================================= #
# --- Extract precipitation --- #
# Extract snowfall and rainfall, and sum
da_snowfall = ds_monthly['var131']
da_rainfall = ds_monthly['var132']
da_prec = da_snowfall + da_rainfall
# Calculate mean annual
da_prec_mean = da_prec.mean(dim='time')
# Conver unit: mm/s -> mm/year (ignore leap years)
da_prec_mean = da_prec_mean * 86400 * 365
da_prec_mean.attrs['unit'] = 'mm/year'

# --- Extract air temperature --- #
da_tair = ds_monthly['var11']
# Calculate mean annual
da_tair_mean = da_tair.mean(dim='time')
# Conver unit: K -> degC
da_tair_mean = da_tair_mean - 273.15
da_tair_mean.attrs['unit'] = 'degC'

# --- Extract total ET --- #
da_ET = ds_monthly['var57']
# Calculate mean annual
da_ET_mean = da_ET.mean(dim='time')
# Conver unit: mm/s -> mm/year
da_ET_mean = da_ET_mean * 86400 * 365
da_ET_mean.attrs['unit'] = 'mm/year'


# Save to file
ds = xr.Dataset({'PREC': da_prec_mean,
                 'AIR_TEMP': da_tair_mean,
                 'ET': da_ET_mean})
ds.to_netcdf(os.path.join(output_agg_dir,
                          'soil_moisture.{}_{}.mean_annual.nc'.format(start_date, end_date)),
             format='NETCDF4_CLASSIC')

