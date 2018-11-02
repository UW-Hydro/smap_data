
import sys
import subprocess
import numpy as np
import os


# ======================================================= #
# Process command line argument
# ======================================================= #
start_date = sys.argv[1]  # YYYYMMDD
end_date = sys.argv[2]  # YYYYMMDD

start_year = int(start_date[:4])
end_year = int(end_date[:4])


# ======================================================= #
# Other parameters
# ======================================================= #
output_dir = '../raw_data'


# ======================================================= #
# Download monthly GLDAS data; VIC
# ======================================================= #
# Loop over each year
for year in range(start_year, end_year+1):
    # --- Make output subdir --- #
    output_year_subdir = "{}/{}".format(output_dir, year)
    if not os.path.exists(output_year_subdir):
        os.makedirs(output_year_subdir)
    # --- Identify url_dirname --- #
    url_dirname = 'hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS_V1/GLDAS_VIC10_M/{}/'.format(year)
    # --- Download data for each month --- #
    for mon in range(1, 13):
        print('Downloading', year, mon)
        # Identify and download the grb file
        grb_filename = "GLDAS_VIC10_M.A{}{:02d}.001.grb".format(year, mon)
        subprocess.call(("wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies "
                         "--keep-session-cookies https://{}{} -O {}/{:02d}.grb").format(
                            url_dirname, grb_filename, output_year_subdir, mon),
                        shell=True)
        # Identify and download the grb.xml file
        grbxml_filename = "GLDAS_VIC10_M.A{}{:02d}.001.grb.xml".format(year, mon)
        subprocess.call(("wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies "
                         "--keep-session-cookies https://{}{} -O {}/{:02d}.grb.xml").format(
                            url_dirname, grbxml_filename, output_year_subdir, mon),
                        shell=True)

