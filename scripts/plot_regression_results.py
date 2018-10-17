
import sys
import xarray as xr
import pickle
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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
output_data_dir = cfg['OUTPUT']['output_dir']


# ======================================= #
# Make output plot dir
# ======================================= #
# Output plot dir
output_plot_dir = os.path.join(output_data_dir, 'plots')
if not os.path.exists(output_plot_dir):
    os.makedirs(output_plot_dir)


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
# Load results dict
with open(os.path.join(
    output_data_dir, result_pickle), 'rb') as f:
    dict_results = pickle.load(f)

# Extract n_coef
for latlon_ind, item in dict_results.items():
    lat_ind = int(latlon_ind.split('_')[0])
    lon_ind = int(latlon_ind.split('_')[1])
    # Extract fitted coef
    fitted_coef = item['model'].coef_
    n_coef = len(fitted_coef)
    break


# ======================================= #
# Extract results
# ======================================= #
print('Extracting results...')

# --- Prepare empty result spatial da --- #
list_coef = []  # A list of results da for each Lasso param
# Prepare an initial da to store results
init = da_domain.values
init = init.astype(float)
init[:] = np.nan
da = da_domain.copy(deep=True)
da = da.astype(float)
da[:] = init
# Fitted coefficients
for i in range(n_coef):
    da_init = da.copy(True)
    list_coef.append(da_init)
# R^2
da_R2 = da.copy(True)
# RMSE
da_RMSE = da.copy(True)

# --- Extract results --- #
for latlon_ind, item in dict_results.items():
    lat_ind = int(latlon_ind.split('_')[0])
    lon_ind = int(latlon_ind.split('_')[1])
    # Extract fitted coef
    fitted_coef = item['model'].coef_
    # Extract R^2
    R2 = item['R2']
    # Extract RMSE
    RMSE = item['RMSE']
    # Put results into da
    for i in range(n_coef):
        list_coef[i][lat_ind, lon_ind] = fitted_coef[i]
    da_R2[lat_ind, lon_ind] = R2
    da_RMSE[lat_ind, lon_ind] = RMSE


# ======================================= #
# Plot R2
# ======================================= #
print('Plotting R2...')

fig = plt.figure(figsize=(12, 5))
# Set projection
ax = plt.axes(projection=ccrs.PlateCarree())
gl = add_gridlines(ax, alpha=0)
# Plot
cs = da_R2.where(da_domain==1).plot.pcolormesh(
    'lon', 'lat', ax=ax,
    add_colorbar=False,
    add_labels=False,
    cmap='plasma_r',
    vmin=0, vmax=1,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='min')
cbar.set_label('R^2', fontsize=20)
plt.title('R^2 (domain-median = {:.2f})'.format(
    da_R2.where(da_domain==1).median().values), fontsize=20)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}R2.png'.format(file_basename)),
    format='png', bbox_inches='tight', pad_inches=0)


# ======================================= #
# Plot RMSE
# ======================================= #
print('Plotting RMSE...')

fig = plt.figure(figsize=(12, 5))
# Set projection
ax = plt.axes(projection=ccrs.PlateCarree())
gl = add_gridlines(ax, alpha=0)
# Plot
cs = da_RMSE.where(da_domain==1).plot.pcolormesh(
    'lon', 'lat', ax=ax,
    add_colorbar=False,
    add_labels=False,
    cmap='plasma_r',
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='min')
cbar.set_label('RMSE (mm/mm * hour-1)', fontsize=20)
plt.title('RMSE (domain-median = {:.2f})'.format(
    da_RMSE.where(da_domain==1).median().values), fontsize=20)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}RMSE.png'.format(file_basename)),
    format='png', bbox_inches='tight', pad_inches=0)


# ======================================= #
# Plot - fitted coefficients
# ======================================= #
print('Ploting fitted coefficients...')

list_coef_name = cfg['PLOT']['coef_names']
list_cbar_label = cfg['PLOT']['cbar_label']
for i in range(n_coef):
    fig = plt.figure(figsize=(12, 5))
    # Set projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    gl = add_gridlines(ax, alpha=0)
    # Determine colormap range - 5th or 95th quantile, whichever
    # has bigger magnitude
    mag1 = np.absolute(list_coef[i].quantile(0.05).values)
    mag2 = np.absolute(list_coef[i].quantile(0.95).values)
    vmax = np.max([mag1, mag2])
    # Plot
    cs = list_coef[i].where(da_domain==1).plot.pcolormesh(
        'lon', 'lat', ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap='Spectral',
        vmin=-vmax, vmax=vmax,
        transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, extend='both')
    cbar.set_label(list_cbar_label[i], fontsize=20)
    plt.title('Fitted coef: '+list_coef_name[i], fontsize=20)
    # Save fig
    fig.savefig(
        os.path.join(
            output_plot_dir,
            '{}fitted_coef.{}.png'.format(file_basename, i+1)),
        format='png', bbox_inches='tight', pad_inches=0)
