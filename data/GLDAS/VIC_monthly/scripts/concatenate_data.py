
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


# ======================================================= #
# Process and subset
# ======================================================= #

list_ds_alltime = []

# Loop over each monthly data
for year in range(start_year, end_year+1):
    # --- Identify subdir for the date --- #
    output_year_subdir = "{}/{}".format(nc_monthly_dir, year)
    # --- Loop over each month --- #
    for mon in range(1, 13):
        print("Loading", year, mon)
        # Identify filename for the monthly data
        nc_monthly = os.path.join(
            output_year_subdir, '{:02d}.nc'.format(mon))
        # Load data
        ds = xr.open_dataset(nc_monthly)
        ds.load()
        ds.close()
        # Put into list
        list_ds_alltime.append(ds)

# Concatenate
print('Concatenating all timesteps...')
ds_concat = xr.concat(list_ds_alltime, dim='time')

# Save to file
print('Saving...')
ds_concat.to_netcdf(os.path.join(nc_monthly_dir,
                                 'GLDAS_VIC.{}_{}.nc'.format(start_date, end_date)),
                    format='NETCDF4_CLASSIC')



