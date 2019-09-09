
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
if not os.path.exists(output_plot_dir):
    os.makedirs(output_plot_dir)

# Output data dir
output_data_dir = os.path.join(output_dir, 'data')
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)


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
# Intercept
if X_version == 'v1_intercept' or X_version == 'v2_intercept':
    da_intercept = xr.open_dataset(
        os.path.join(output_data_dir,
        '{}fitted_intercept.nc'.format(file_basename)))['intercept']
    da_intercept.load()
    da_intercept.close()
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
# Plot R2
# ======================================= #
print('Plotting R2...')

fig = plt.figure(figsize=(12, 5))
# Set projection
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND,
               facecolor=[1, 1, 1],
               edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
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
cbar.set_label('$R^2$', fontsize=20)
#plt.title('$R^2$', fontsize=20)
# Insert pdf plot
a = plt.axes([0.16, 0.3, 0.12, 0.2])
data_all = da_R2.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, density=True, color='gray', range=(0, 1))
plt.xlabel('$R^2$', fontsize=14)
plt.title('PDF', fontsize=14)
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
ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND,
               facecolor=[1, 1, 1],
               edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
gl = add_gridlines(ax, alpha=0)
# Plot
cs = da_RMSE.where(da_domain==1).plot.pcolormesh(
    'lon', 'lat', ax=ax,
    add_colorbar=False,
    add_labels=False,
    cmap='plasma_r',
    vmin=0, vmax=0.005,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='max')
cbar.set_label('RMSE (mm/mm / hour)', fontsize=20)
#plt.title('RMSE (domain-median = {:.4f})'.format(
#    da_RMSE.where(da_domain==1).median().values), fontsize=20)
# Insert pdf plot
a = plt.axes([0.16, 0.3, 0.12, 0.2])
data_all = da_RMSE.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, density=True, color='gray')
plt.xlabel('RMSE\n(mm/mm / hour)', fontsize=14)
plt.title('PDF', fontsize=14)
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

# --- Plot raw coefficients --- #
for i in range(n_coef):
    fig = plt.figure(figsize=(12, 5))
    # Set projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    gl = add_gridlines(ax, alpha=0)
    ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
    ax.add_feature(cartopy.feature.LAND,
               facecolor=[1, 1, 1],
               edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
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
    cbar.set_label(r'$\beta${}'.format(i+1), fontsize=20)
    plt.title(r'Fitted coef $\beta${}'.format(i+1), fontsize=20)
    # Insert pdf plot
    a = plt.axes([0.16, 0.3, 0.12, 0.2])
    data_all = list_coef[i].values.flatten()
    data_all = data_all[~np.isnan(data_all)]
    cs = plt.hist(data_all, bins=20, density=True, color='gray')
    plt.xlabel(r'$\beta${}'.format(i+1), fontsize=14)
    plt.title('PDF', fontsize=14)
    # Save fig
    fig.savefig(
        os.path.join(
            output_plot_dir,
            '{}fitted_coef.{}.png'.format(file_basename, i+1)),
        format='png', bbox_inches='tight', pad_inches=0)


# --- Interpretation of beta 1 (SM) --- #
# 1) -beta1 - linear loss function slope
beta1 = - list_coef[0] * 24  # convert to [day-1]
fig = plt.figure(figsize=(12, 5))
# Set projection
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
gl = add_gridlines(ax, alpha=0)
ax.add_feature(cartopy.feature.LAND,
               facecolor=[1, 1, 1],
               edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
# Plot
cs = beta1.where(da_domain==1).plot.pcolormesh(
    'lon', 'lat', ax=ax,
    add_colorbar=False,
    add_labels=False,
    cmap='Spectral_r',
    vmin=0, vmax=0.6,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='both')
cbar.set_label(r'${\beta}_1$ ' + r'$(day^{-1})$', fontsize=20)
#plt.title('Loss function linear slope', fontsize=20)
# Insert pdf plot
a = plt.axes([0.16, 0.3, 0.12, 0.2])
data_all = beta1.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, range=(0, 0.6),
              density=True, color='gray')
plt.xlabel(r'${\beta}_1$ ' + r'$(day^{-1})$', fontsize=14)
plt.title('PDF', fontsize=14)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}.loss_slope.png'.format(file_basename, i+1)),
    format='png', bbox_inches='tight', pad_inches=0)

# 2) -1/beta1 - exponential decay e-folding time scale tau
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
    cmap='Spectral',
    vmin=0, vmax=20,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='both')
cbar.set_label(r'${\tau}$ ' + '(day)', fontsize=20)
#plt.title('SM exponential decay e-folding time scale', fontsize=20)
# Insert pdf plot
a = plt.axes([0.17, 0.3, 0.12, 0.2])
data_all = tau.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, range=(0, 30),
              density=True, color='gray')
plt.ylim([0, 0.3])
plt.xlabel(r'${\tau}$ ' + '(day)', fontsize=14)
plt.title('PDF', fontsize=14)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}.decay_tau.png'.format(file_basename, i+1)),
    format='png', bbox_inches='tight', pad_inches=0)


# --- Interpretation of beta 2 (P) --- #
# beta2 * depth [mm] - fraction of P flux that is added to the top 5cm soil
# (if P*SM presents, this interpretation is for when SM=0)
P_frac = list_coef[1] * cfg['PLOT']['soil_depth']  # convert from [mm-1] to [-]
# Set up label
if X_version[:2] == 'v1':
    label = r'${\beta}_2$ ' + '(-)'
elif X_version[:2] == 'v2':
    label = r'${\gamma}_2$ ' + '(-)'

# Plot
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
    cmap='Spectral',
    vmin=0, vmax=1,
    transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, extend='both')
cbar.set_label(label, fontsize=20)
#plt.title('Fraction of P flux reflected in the surface-layer SM\n'
#          '(if P*SM presents, this interpretation is for when SM=0)',
#          fontsize=20)
# Insert pdf plot
a = plt.axes([0.16, 0.3, 0.12, 0.2])
data_all = P_frac.values.flatten()
data_all = data_all[~np.isnan(data_all)]
cs = plt.hist(data_all, bins=20, range=(0, 1.5),
              density=True, color='gray')
plt.ylim([0, 4])
plt.xlabel(label, fontsize=14)
plt.title('PDF', fontsize=14)
# Save fig
fig.savefig(
    os.path.join(
        output_plot_dir,
        '{}.P_frac.png'.format(file_basename, i+1)),
    format='png', bbox_inches='tight', pad_inches=0)


# --- Interpretation of beta 3 (SM*P) --- #
# beta3 - how much P_frac changes with SM level
# only for X_v2 or X_v3
if X_version == 'v2' or X_version == 'v2_intercept':
    P_frac_with_SM = list_coef[2]  # convert from [-/mm]
    fig = plt.figure(figsize=(12, 5))
    # Set projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
    gl = add_gridlines(ax, alpha=0)
    ax.add_feature(cartopy.feature.LAND,
                   facecolor=[1, 1, 1],
                   edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
    # Plot
    cs = P_frac_with_SM.where(da_domain==1).plot.pcolormesh(
        'lon', 'lat', ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap='Spectral',
        vmin=-0.2, vmax=0,
        transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, extend='both')
    cbar.set_label(r'${\gamma}_3$ ' + '(-/mm)',
                   fontsize=20)
#    plt.title('Sensitivity of fraction of P flux (reflected \n'
#              'in the surface-layer SM) to SM level',
#              fontsize=20)
    # Insert pdf plot
    a = plt.axes([0.16, 0.3, 0.12, 0.2])
    data_all = P_frac_with_SM.values.flatten()
    data_all = data_all[~np.isnan(data_all)]
    cs = plt.hist(data_all, bins=20, range=(-0.5, 0),
                  density=True, color='gray')
    plt.ylim([0, 30])
    plt.xlabel(r'${\gamma}_3$ ' + '(-/mm)', fontsize=14)
    plt.title('PDF', fontsize=14)
    # Save fig
    fig.savefig(
        os.path.join(
            output_plot_dir,
            '{}.P_frac_with_SM.png'.format(file_basename, i+1)),
        format='png', bbox_inches='tight', pad_inches=0)


# --- Interpretation of beta 0 (intercept) --- #
if X_version == 'v1_intercept' or X_version == 'v2_intercept':
    # SM0 = -beta0/beta1 - fitted lowest SM0
    sm0 = - da_intercept / list_coef[0]  # [-]
    fig = plt.figure(figsize=(12, 5))
    # Set projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -85, 85], ccrs.Geodetic())
    gl = add_gridlines(ax, alpha=0)
    ax.add_feature(cartopy.feature.LAND,
                   facecolor=[1, 1, 1],
                   edgecolor=[0.5, 0.5, 0.5], linewidth=0.3)
    # Plot
    cs = sm0.where(da_domain==1).plot.pcolormesh(
        'lon', 'lat', ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap='Spectral',
        vmin=0, vmax=0.5,
        transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, extend='both')
    cbar.set_label(r'${SSM_0}$ ' + r'$(mm^{3}/mm^{3})$', fontsize=20)
    #plt.title('SM exponential decay e-folding time scale', fontsize=20)
    # Insert pdf plot
    a = plt.axes([0.17, 0.3, 0.12, 0.2])
    data_all = sm0.values.flatten()
    data_all = data_all[~np.isnan(data_all)]
    cs = plt.hist(data_all, bins=20, range=(0, 0.5),
                  density=True, color='gray')
    plt.ylim([0, 10])
    plt.xlabel(r'${SSM_0}$ ' + r'$(mm^{3}/mm^{3})$', fontsize=14)
    plt.title('PDF', fontsize=14)
    # Save fig
    fig.savefig(
        os.path.join(
            output_plot_dir,
            '{}.sm0.png'.format(file_basename, i+1)),
        format='png', bbox_inches='tight', pad_inches=0)
