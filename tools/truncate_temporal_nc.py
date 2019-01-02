
import xarray as xr
import pandas as pd
import sys


# --- Read command line arguments --- #
input_nc = sys.argv[1]
truncate_start_time = sys.argv[2]
truncate_end_time = sys.argv[3]
output_nc = sys.argv[4]

# --- Load input nc --- #
print('Loading file...')
ds = xr.open_dataset(input_nc)
ds.load()
ds.close()

# --- Truncate temporally --- #
print('Truncating...')
ds_new = ds.sel(time=slice(truncate_start_time, truncate_end_time))

# --- Save to file --- #
print('Saving...')
ds_new.to_netcdf(output_nc, format='NETCDF4_CLASSIC')

