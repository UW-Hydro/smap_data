
# Construct SMAP global domain file from SMAP processed data

import xarray as xr
import numpy as np


# Load processed SMAP data
ds = xr.open_dataset('/pool0/data/yixinmao/smap_data/tools/prepare_SMAP/output/data/soil_moisture.20150331_20171231.nc')

# Calculate temporal mean map - this will be used as the SMAP domain file
# NOTE: this step sets all pixels with ANY non-NAN timesteps to be in the domain
da_smap_mean = ds['soil_moisture'].mean(dim='time')

# Construct domain array
da_domain = da_smap_mean.copy()
domain = da_domain[:].values
domain[~np.isnan(domain)] = 1
domain[np.isnan(domain)] = 0
domain = domain.astype(int)

# Save domain to netCDF file
da_domain = xr.DataArray(domain,
                       coords=[da_smap_mean['lat'].values,
                               da_smap_mean['lon'].values],
                       dims=['lat', 'lon'])
ds_domain = xr.Dataset({'mask': da_domain})
ds_domain.to_netcdf('/pool0/data/yixinmao/smap_data/param/domain/smap.domain.global.nc',
                    format='NETCDF4_CLASSIC')


