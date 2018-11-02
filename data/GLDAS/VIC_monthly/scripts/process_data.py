
import sys
import subprocess
import pandas as pd
import os
import numpy as np


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
raw_data_dir = '../raw_data'
output_dir = '../monthly_nc'


# ======================================================= #
# Process and subset
# ======================================================= #

# Loop over each monthly data
for year in range(start_year, end_year+1):
    # --- Make output subdir --- #
    output_year_subdir = "{}/{}".format(output_dir, year)
    if not os.path.exists(output_year_subdir):
        os.makedirs(output_year_subdir)

    # --- Loop over each month --- #
    for mon in range(1, 13):
        print("Processing", year, mon)
        # Convert grb to netCDF
        grb_path = os.path.join(
            raw_data_dir,
            "{}".format(year),
            "{:02d}.grb".format(mon))
        output_nc = os.path.join(
            output_year_subdir, '{:02d}.allvars.nc'.format(mon))
        subprocess.call("cdo -f nc copy {} {}".format(grb_path, output_nc), shell=True)

        # Only keep needed variables and delete the rest to reduce size
        output_final_nc = os.path.join(
            output_year_subdir, '{:02d}.nc'.format(mon))
        subprocess.call(
            "ncks -O -v var131,var132,var65,var86,var11,var57 {} {}".format(
                output_nc, output_final_nc),
            shell=True)
        os.remove(output_nc)


