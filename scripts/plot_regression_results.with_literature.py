
import sys
import xarray as xr
import pickle
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import subprocess

from tonic.io import read_config, read_configobj
from sm_utils import add_gridlines


# ========================================= #
# Process command line arguments
# ========================================= #
cfg = read_configobj(sys.argv[1])


# ======================================= #
# Parameter setting
# ======================================= #
# Output data dir
output_dir = cfg['OUTPUT']['output_dir']


# ======================================= #
# Make output plot dir
# ======================================= #
# Output plot dir
output_plot_dir = os.path.join(output_dir, 'plots')
subprocess.call("mkdir -p {}".format(output_plot_dir), shell=True)

# Output data dir
output_data_dir = os.path.join(output_dir, 'data')
subprocess.call("mkdir -p {}".format(output_data_dir), shell=True)


# ======================================= #
# Load data
# ======================================= #
print('Loading results...')

# --- Domain --- #
da_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])['mask']

# --- Regression results --- #
# Extract regression information
regression_type = cfg['REGRESSION']['regression_type']
standardize = cfg['REGRESSION']['standardize']
X_version = cfg['REGRESSION']['X_version']
if regression_type == 'linear':
    pass
elif regression_type == 'lasso' or regression_type == 'ridge':
    alpha = cfg['REGRESSION']['alpha']
# Construct result filename
file_basename = 'X_{}.{}.{}.{}'.format(
    X_version,
    regression_type,
    'standardize' if standardize else 'no_standardize',
    '{}.'.format(alpha) if regression_type == 'lasso' or regression_type == 'ridge' else '')
result_pickle = 'results.{}pickle'.format(file_basename)

# Determine n_coef
if X_version == 'v1' or X_version == 'v1_intercept':
    n_coef = 2
elif X_version == 'v2' or X_version == 'v2_intercept':
    n_coef = 3
elif X_version == 'v3':
    n_coef = 4
elif X_version == 'v4':
    n_coef = 3


# ======================================= #
# Load result netCDF
# ======================================= #
print('Loading results...')

# Coef
list_coef = []
for i in range(n_coef):
    da_coef = xr.open_dataset(
        os.path.join(output_data_dir,
        '{}fitted_coef.{}.nc'.format(file_basename, i+1)))['coef']
    da_coef.load()
    da_coef.close()
    list_coef.append(da_coef)
# R2
da_R2 = xr.open_dataset(os.path.join(
    output_data_dir,
    '{}R2.nc'.format(file_basename, i+1)))['R2']
da_R2.load()
da_R2.close()
# RMSE
da_RMSE = xr.open_dataset(os.path.join(
    output_data_dir,
    '{}RMSE.nc'.format(file_basename, i+1)))['RMSE']
da_RMSE.load()
da_RMSE.close()


# ======================================= #
# Plot - fitted coefficients
# ======================================= #
print('Ploting fitted coefficients...')

# --- Interpretation of beta 1 (SM) - tau, compare with McColl et al. [2017b] --- #
# -1/beta1 - exponential decay e-folding time scale tau
tau = - 1 / (list_coef[0] * 24)  # [day]
fig = plt.figure(figsize=(12, 5))
# Set projection
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
gl = add_gridlines(ax, alpha=0)
ax.add_feature(cartopy.feature.LAND,
               facecolor=[1, 1, 1],
               edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
# Plot
cs = tau.where(da_domain==1).plot.pcolormesh(
    'lon', 'lat', ax=ax,
    add_colorbar=False,
    add_labels=False,
    cmap='YlGnBu',
    vmin=0, vmax=20,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='both')
cbar.set_label(r'${\tau}$ ' + '(day)', fontsize=20)
#plt.title('SM exponential decay e-folding time scale', fontsize=20)
# Insert pdf plot
a = plt.axes([0.17, 0.3, 0.12, 0.2])
data_all = tau.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, range=(0, 40),
              density=True, color='gray')
plt.xlabel(r'${\tau}$ ' + '(day)', fontsize=14)
plt.title('PDF', fontsize=14)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}.decay_tau.with_McColl_2017b.png'.format(file_basename, i+1)),
    format='png', bbox_inches='tight', pad_inches=0)


# --- Interpretation of beta 2 (P) - with McColl et al. [2017a] --- #
# beta2 * depth [mm] - fraction of P flux that is added to the top 5cm soil
# (if P*SM presents, this interpretation is for when SM=0)
P_frac = list_coef[1] * cfg['PLOT']['soil_depth']  # convert from [mm-1] to [-]
fig = plt.figure(figsize=(12, 5))
# Set projection
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
gl = add_gridlines(ax, alpha=0)
ax.add_feature(cartopy.feature.LAND,
               facecolor=[1, 1, 1],
               edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
# Plot
cs = P_frac.where(da_domain==1).plot.pcolormesh(
    'lon', 'lat', ax=ax,
    add_colorbar=False,
    add_labels=False,
    cmap='cool',
    vmin=0, vmax=1,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='both')
cbar.set_label(r'${\beta}_2$ ' + '(-)', fontsize=20)
#plt.title('Fraction of P flux reflected in the surface-layer SM\n'
#          '(if P*SM presents, this interpretation is for when SM=0)',
#          fontsize=20)
# Insert pdf plot
a = plt.axes([0.16, 0.3, 0.12, 0.2])
data_all = P_frac.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, range=(0, 1),
              density=True, color='gray')
plt.xlabel(r'${\beta}_2$ ' + '(-)', fontsize=14)
plt.title('PDF', fontsize=14)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}.P_frac.with_McColl_2017a.png'.format(file_basename, i+1)),
    format='png', bbox_inches='tight', pad_inches=0)


# --- Interpretation of beta 2 (P) - with Akbar et al. [2018b] --- #
# 1 / beta2  - effective depth delta z
# (if P*SM presents, this interpretation is for when SM=0)
dz = 1 / list_coef[1]  # convert from [mm-1] to [mm]
# Extract CONUS
dz = dz.sel(lat=slice(25, 50), lon=slice(-125, -60))
fig = plt.figure(figsize=(12, 5))
# Set projection
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-125, -60, 25, 50], ccrs.PlateCarree())
gl = add_gridlines(ax,
                   xlocs=[-120, -100, -80, -60],
                   ylocs=[30, 40, 50],
                   alpha=0)
ax.add_feature(cartopy.feature.LAND,
               facecolor=[1, 1, 1],
               edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
# Plot
cs = dz.where(da_domain==1).plot.pcolormesh(
    'lon', 'lat', ax=ax,
    add_colorbar=False,
    add_labels=False,
    cmap='Spectral_r',
    vmin=0, vmax=400,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='both')
cbar.set_label(r'$\Delta$z [mm]', fontsize=20)
#plt.title('Fraction of P flux reflected in the surface-layer SM\n'
#          '(if P*SM presents, this interpretation is for when SM=0)',
#          fontsize=20)
# Insert pdf plot
a = plt.axes([0.6, 0.32, 0.12, 0.2])
data_all = dz.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, range=(0, 400),
              density=True, color='gray')
plt.xlabel(r'$\Delta$z [mm]', fontsize=14)
plt.title('PDF', fontsize=14)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}.delta_z.with_Akbar_2018b.png'.format(file_basename, i+1)),
    format='png', bbox_inches='tight', pad_inches=0)


