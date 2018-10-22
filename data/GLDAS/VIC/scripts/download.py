
import sys
import subprocess
import pandas as pd
import numpy as np
import re
import os


# ======================================================= #
# Process command line argument
# ======================================================= #
start_date = sys.argv[1]  # YYYYMMDD
end_date = sys.argv[2]  # YYYYMMDD

# ======================================================= #
# Other parameters
# ======================================================= #
output_dir = '../raw_data'

# ======================================================= #
# Download 3-hourly GLDAS-2 data; CLM
# ======================================================= #
dates = pd.date_range(start_date, end_date, freq='1D')
for date in dates:
    day_of_year = date.dayofyear
    # --- Make output subdir --- #
    output_date_subdir = "{}/{}/{:02d}/{:02d}".format(output_dir, date.year, date.month, date.day)
    if not os.path.exists(output_date_subdir):
        os.makedirs(output_date_subdir)

    # --- Identify all filenames of the day --- #
    # Download the html file
    url_dirname = 'hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS_V1/GLDAS_VIC10_3H/{}/{:03d}/'.format(date.year, day_of_year)
#    subprocess.call(("wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies "
#                     "--keep-session-cookies "
#                     "https://{} -O {}{:03d}.html").format(url_dirname, date.year, day_of_year),
#                    shell=True)
#    # Load in the html file
#    with open('{}{:03d}.html'.format(date.year, day_of_year), 'r') as f:
#        html_lines = f.read()

    # --- Download data for each 3-hour --- #
    for hour in np.arange(0, 24, 3):
        print('Downloading', date, hour)
        # Identify and download the grb file
        grb_filename = "GLDAS_VIC10_3H.A{}{:03d}.{:04d}.001.grb".format(date.year, day_of_year, hour*100)
        subprocess.call(("wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies "
                         "--keep-session-cookies https://{}{} -O {}/{:04d}.grb").format(
                            url_dirname, grb_filename, output_date_subdir, hour*100),
                        shell=True)
        # Identify and download the grb.xml file
        grbxml_filename = "GLDAS_VIC10_3H.A{}{:03d}.{:04d}.001.grb.xml".format(date.year, day_of_year, hour*100)
        subprocess.call(("wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies "
                         "--keep-session-cookies https://{}{} -O {}/{:04d}.grb.xml").format(
                            url_dirname, grbxml_filename, output_date_subdir, hour*100),
                        shell=True)

