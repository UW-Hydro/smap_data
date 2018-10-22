
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


# ======================================================= #
# Process and subset
# ======================================================= #

list_ds_alltime = []

# Loop over each 3-hourly data
dates = pd.date_range(start_date, end_date, freq='1D')
for t, date in enumerate(dates):
    # --- Identify subdir for the date --- #
    output_date_subdir = "{}/{}/{:02d}/{:02d}".format(nc_3hourly_dir, date.year, date.month, date.day)
    
    # --- Loop over each 3-hour --- #
    for hour in np.arange(0, 24, 3):
        print("Loading", date, hour)
        # Identify filename for the 3-hour data
        nc_3hourly = os.path.join(
            output_date_subdir, '{:02d}.nc'.format(hour))
        # Load data
        ds = xr.open_dataset(nc_3hourly)
        ds.load()
        ds.close()
        # Put into list
        list_ds_alltime.append(ds)

# Concatenate
print('Concatenating all timesteps...')
ds_concat = xr.concat(list_ds_alltime, dim='time')

# Save to file
print('Saving...')
ds_concat.to_netcdf(os.path.join(nc_3hourly_dir,
                                 'GLDAS_VIC.{}_{}.nc'.format(start_date, end_date)),
                    format='NETCDF4_CLASSIC')



