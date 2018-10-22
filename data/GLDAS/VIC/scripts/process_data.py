
import sys
import subprocess
import pandas as pd
import os
import numpy as np


# ======================================================= #
# Process command line argument
# ======================================================= #
start_date = sys.argv[1]  # YYYYMMDD
end_date = sys.argv[2]  # YYYYMMDD


# ======================================================= #
# Other parameters
# ======================================================= #
# Paths
raw_data_dir = '../raw_data'
output_dir = '../3hourly_nc'

# ======================================================= #
# Process and subset
# ======================================================= #

# Loop over each 3-hourly data
dates = pd.date_range(start_date, end_date, freq='1D')
for t, date in enumerate(dates):
    # --- Make subdir for the date --- #
    output_date_subdir = "{}/{}/{:02d}/{:02d}".format(output_dir, date.year, date.month, date.day)
    if not os.path.exists(output_date_subdir):
        os.makedirs(output_date_subdir)
    
    # --- Loop over each 3-hour --- #
    for hour in np.arange(0, 24, 3):
        print("Processing", date, hour)
        # Convert grb to netCDF
        grb_path = os.path.join(
            raw_data_dir,
            "{}".format(date.year),
            "{:02d}".format(date.month),
            "{:02d}".format(date.day),
            "{:04d}.grb".format(hour*100))
        output_nc = os.path.join(
            output_date_subdir, '{:02d}.allvars.nc'.format(hour))
        subprocess.call("cdo -f nc copy {} {}".format(grb_path, output_nc), shell=True)

        # Only keep needed variables and delete the rest to reduce size
        output_final_nc = os.path.join(
            output_date_subdir, '{:02d}.nc'.format(hour))
        subprocess.call(
            "ncks -O -v var131,var132,var65,var86 {} {}".format(
                output_nc, output_final_nc),
            shell=True)
        os.remove(output_nc)


