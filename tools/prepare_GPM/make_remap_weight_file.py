
# This script loads and process the raw downloaded GPM IMERG data


import sys
import pandas as pd
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tonic.io import read_config, read_configobj
from prep_forcing_utils import (to_netcdf_forcing_file_compress, setup_output_dirs,
                                make_weight_file)


# ======================================================= #
# Process command line argument
# ======================================================= #
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Setup output subdirs
# ============================================================ #
output_dir = cfg['OUTPUT']['out_dir']

output_subdir_plots = setup_output_dirs(output_dir, mkdirs=['plots'])['plots']
output_subdir_data = setup_output_dirs(
    output_dir, mkdirs=['data'])['data']
output_subdir_tmp = setup_output_dirs(output_dir, mkdirs=['tmp'])['tmp']


# ======================================================= #
# Process time index
# ======================================================= #
start_time = pd.to_datetime(cfg['TIME']['start_time'])
end_time = pd.to_datetime(cfg['TIME']['end_time'])
start_year = start_time.year
end_year = end_time.year


# ======================================================= #
# Generate remap weight file, if not input
# ======================================================= #
# --- Load the first GPM timestep --- #
filename = os.path.join(
    cfg['GPM']['gpm_dir'],
    '{}'.format(start_time.year), '{:02d}'.format(start_time.month),
    '{:02d}'.format(start_time.day),
    '{}.{:02d}00.nc'.format(start_time.strftime('%Y%m%d'), start_time.hour))
da = xr.open_dataset(filename)['precipitationCal']
da = da.transpose('lat', 'lon')
# --- Remap to the target domain --- #
# Load target domain file
ds_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])
da_domain = ds_domain[cfg['DOMAIN']['mask_name']]
# Extract GPM domain
da_gpm_domain = xr.DataArray(np.ones([len(da['lat']), len(da['lon'])],
                                     dtype=int),
                             coords=[da['lat'], da['lon']],
                             dims=['lat', 'lon'])
# Remap to get weight file
if 'REMAP' in cfg and 'weight_nc' in cfg['REMAP']:
    final_weight_nc = cfg['REMAP']['weight_nc']
else:
    reuse_weight = False
    final_weight_nc = os.path.join(output_subdir_tmp, 'gpm_to_vic_weights.nc')
    A = make_weight_file(
        reuse_weight=False,
        da_source=da,
        final_weight_nc=final_weight_nc,
        da_source_domain=da_gpm_domain,
        da_target_domain=da_domain,
        tmp_weight_nc=os.path.join(output_subdir_tmp, 'gpm_to_vic_weights.tmp.nc'),
        process_method=None)

