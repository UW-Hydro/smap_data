
import xarray as xr
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
import scipy.stats
import sys

from tonic.io import read_config, read_configobj
from sm_utils import add_gridlines


# ========================================= #
# Process command line arguments
# ========================================= #
# regresion cfg file
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
    '{}.'.format(lasso_alpha) if regression_type == 'lasso' else '')
# Determine n_coef
if X_version == 'v1':
    n_coef = 2
elif X_version == 'v2':
    n_coef = 3
elif X_version == 'v3':
    n_coef = 4
elif X_version == 'v4':
    n_coef = 3

# --- Load results --- #
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


# --- Load input SMAP and GPM data --- #
# Domain
da_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])['mask']
# SMAP (instantaneous time points)
da_smap = xr.open_dataset(cfg['INPUT']['smap_nc'])['soil_moisture']
# GPM (time-beginning timestamp)
da_prec = xr.open_dataset(cfg['INPUT']['gpm_nc'])['PREC']

da_smap.load()
da_smap.close()
da_prec.load()
da_prec.close()


# ======================================= #
# Extract pixel to investigate
# ======================================= #
# --- Sahara --- #
# 1) with very little P; P and SM*P highly correlated
lat_ind = 281
lon_ind = 555

# --- Extract time series --- #
ts_smap = da_smap[:, lat_ind, lon_ind].to_series()
ts_prec = da_prec[:, lat_ind, lon_ind].to_series()

# --- Regression for the pixel --- #
# Put SMAP and prec into a df
ts_prec = ts_prec.truncate(before=ts_smap.index[0],
                           after=ts_smap.index[-1])
ts_smap = ts_smap.truncate(before=ts_prec.index[0],
                           after=ts_prec.index[-1])
df = pd.concat([ts_smap, ts_prec], axis=1,
               keys=['sm', 'prec'])

# --- Construct X and Y for regression --- #
# Find all time indices with SMAP observations
smap_ind = np.argwhere(~np.isnan(df['sm']))[:, 0]
# Construct X and Y matrices
X = []
Y = []
times = []
# Start from the second SMAP obs
for i in range(1, len(smap_ind)):
    # --- If there the SMAP data gap is too big, skip calculation --- #
    # (Right now set the limit to 5 days)
    if smap_ind[i] - smap_ind[i-1] > 10:
        continue
    # --- Discard this timestep if precipitation data contains NAN
    # Calculate cumulative precipitation (NOTE: prec is time-beginning timestamp
    prec_sum = df.iloc[smap_ind[i-1]:smap_ind[i], :]['prec'].sum()  # [mm]
    if np.isnan(prec_sum) is True:  # Discard timesteps with NAN precipitation data
        continue
    # --- Calculate Y and X elements --- #
    dt = (smap_ind[i] - smap_ind[i-1]) * 12  # delta t [hour]
    # Calculate y
    sm = df.iloc[smap_ind[i], :]['sm']
    sm_last = df.iloc[smap_ind[i-1], :]['sm']
    y = (sm - sm_last) / dt  # [mm/hour]
    Y.append(y)
    # Calculate x
    prec = prec_sum / dt  # [mm/hour]
    if X_version == 'v1':
        x = [sm_last, prec]
    elif X_version == 'v2':
        x = [sm_last, prec, sm_last * prec]
    X.append(x)
    # Save time
    times.append(df.index[smap_ind[i]])
Y = np.asarray(Y)
X = np.asarray(X)
times = pd.to_datetime(times)

# --- Plot pairwise scatterplot for Y and X --- #
sns.set(style="ticks", color_codes=True)
for i in range(n_coef-1):
    for j in range(i+1, n_coef):
        print('Corrcoef of {} and {}:'.format(i, j),
              np.corrcoef(X[:, i], X[:, j])[0, 1])
# Pairwise scatterplot
# Useful source: https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
X_to_plot = X.copy()
Y_to_plot = Y.copy()
Y_to_plot[:] = Y_to_plot[:] * 50 * 24 # convert [mm/mm/hour] to [mm/day]
X_to_plot[:, 0] = X_to_plot[:, 0] * 50  # convert [mm/mm] to [mm]
X_to_plot[:, 2] = X_to_plot[:, 2] * 50 * 24  # convert [mm/hour] to [mm2/day]
df_YX = pd.DataFrame(
    np.concatenate((Y_to_plot.reshape([len(Y), 1]), X_to_plot), axis=1),
    columns=[r'$\Delta$SSM/$\Delta$t (mm/day)', 'SSM (mm)',
             'P (mm/day)', 'SSM*P ($mm^2$/day)'])
# Create an instance of the PairGrid class.
#grid = sns.PairGrid(data=df_YX, size=2)
sns_plot = sns.pairplot(df_YX, size=2)
# Save fig
sns_plot.savefig(
    os.path.join(
        output_plot_dir,
        'scatterpair.{}_{}.png'.format(lat_ind, lon_ind)),
    format='png', dpi=150,
    bbox_inches='tight', pad_inches=0)

print(da_smap['lat'][lat_ind].values)
print(da_smap['lon'][lon_ind].values)

