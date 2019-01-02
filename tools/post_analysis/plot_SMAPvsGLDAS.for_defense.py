
import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import NearestNeighbors

from tonic.io import read_config, read_configobj
from utils import calc_monthly_PET, find_1deg_grid_gldas, edges_from_centers


# ========================================= #
# Parameter setting
# ========================================= #
# Load both SMAP and GLDAS, v1 and v2 results
dict_cfg = {}
for v in ['v1', 'v2']:
    dict_cfg['SMAP_{}'.format(v)] = read_configobj(
        '/civil/hydro/ymao/smap_data/scripts/cfg/20181121.SMAP_recom_qual.exclude_arid/'
        'X_{}.linear.cfg'.format(v))
    dict_cfg['GLDAS_{}'.format(v)] = read_configobj(
        '/civil/hydro/ymao/smap_data/scripts/cfg/20181121.GLDAS_VIC.exclude_arid/smap_freq/'
        'X_{}.linear.cfg'.format(v))
    
# Output data dir
output_dir = 'output/20181121.smap_gldas/for_defense'

# --- Load GLDAS monthly data --- #
ds_monthly = xr.open_dataset(
    '/civil/hydro/ymao/smap_data/data/GLDAS/VIC_monthly/aggregated_variables/soil_moisture.197901_201712.monthly.nc')
ds_monthly.load()
ds_monthly.close()


# ======================================= #
# Make output plot dir
# ======================================= #
# Output plot dir
output_plot_dir = os.path.join(output_dir, 'plots')
if not os.path.exists(output_plot_dir):
    os.makedirs(output_plot_dir)
    
# Output data dir - for loading results
dict_output_data_dir = {}
for key in dict_cfg:
    cfg = dict_cfg[key]
    dict_output_data_dir[key] = os.path.join(
        cfg['OUTPUT']['output_dir'], 'data')


# ======================================= #
# Load data
# ======================================= #
print('Loading results...')

# --- Domain --- #
dict_da_domain = {}  # {SMAP/GLDAS: da}
for key in ['SMAP', 'GLDAS']:
    cfg = dict_cfg['{}_v1'.format(key)]
    dict_da_domain[key] = xr.open_dataset(
        cfg['DOMAIN']['domain_nc'])['mask']

dict_list_coef = {}
for key in dict_cfg:
    print(key)
    cfg = dict_cfg[key]
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
            os.path.join(dict_output_data_dir[key],
            '{}fitted_coef.{}.nc'.format(file_basename, i+1)))['coef']
        da_coef.load()
        da_coef.close()
        list_coef.append(da_coef)
    
    # --- Put final results into dict --- #
    dict_list_coef[key] = list_coef


# ======================================= #
# Calculate climatology
# ======================================= #
# --- Climatology precipitation --- #
da_clim_prec = ds_monthly['PREC'].mean(dim='time') * 12  # [mm/year]
# --- Climatology air T --- #
da_clim_airT = ds_monthly['AIR_TEMP'].mean(dim='time')  # [degC]
# --- Climatology PET - estimated by Equation 7-64 in Dingman --- #
# Climatology monthly airT
da_airT_clim_mon = ds_monthly['AIR_TEMP'].groupby(
    'time.month').mean(dim='time')  # 12 values [degC]
# Calculate climatology monthly PET
da_clim_PET_mon = calc_monthly_PET(da_airT_clim_mon)  # [mm/month]
# Calculate climatology annual PET
da_clim_PET = da_clim_PET_mon.mean(dim='month') * 12  # [mm/year]
# --- Climatology air ET --- #
da_clim_ET = ds_monthly['ET'].mean(dim='time') * 12  # [mm/year]
# --- Aridity index (PET/P) --- #
da_aridity = da_clim_PET / da_clim_prec


# ======================================= #
# Map aridity to SMAP grid
# ======================================= #
# Convert GLDAS aridity to XYZ form
lats_gldas = da_aridity['lat'].values
lons_gldas = da_aridity['lon'].values
lonlon_gldas, latlat_gldas = np.meshgrid(lons_gldas, lats_gldas)
x_gldas = lonlon_gldas.flatten()
y_gldas = latlat_gldas.flatten()
aridity = da_aridity.values.flatten()
# Convert SMAP grid to XY form
lats_smap = dict_da_domain['SMAP']['lat'].values
lons_smap = dict_da_domain['SMAP']['lon'].values
lonlon_smap, latlat_smap = np.meshgrid(lons_smap, lats_smap)
x_smap = lonlon_smap.flatten()
y_smap = latlat_smap.flatten()
# Fit nearest neighbors to SMAP pixels
X = np.stack([x_gldas, y_gldas], axis=1)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
X_smap = np.stack([x_smap, y_smap], axis=1)
distances, indices = nbrs.kneighbors(X_smap)
# Map GLDAS aridity to SMAP pixels
aridity_smap = aridity[indices[:, 0]].reshape(
    [len(dict_da_domain['SMAP']['lat']),
     len(dict_da_domain['SMAP']['lon'])])
da_aridity_smap = xr.DataArray(
    aridity_smap,
    coords=[dict_da_domain['SMAP']['lat'],
            dict_da_domain['SMAP']['lon']],
    dims=['lat', 'lon'])
# --- Put in dict --- #
dict_da_aridity = {'GLDAS': da_aridity,
                   'SMAP': da_aridity_smap}


# ======================================= #
# Plot
# ======================================= #
# Soil depth [mm]
soil_depth = {'SMAP': 50,
              'GLDAS': 100}
# ylim for each coef
ylim = {0: [0, 100],
        1: [0, 1.0],
        2: [-0.2, 0.02]}
# ylabel for each coef
ylabel = {0: r'$\tau$ (day)',
          1: r'$\beta_2$ (-)',
          2: r'$\gamma_3$ (-/mm)'}
# color for SMAP/GLDAS
color = {'SMAP': 'darkorange',
         'GLDAS': '#375E97'}
# whether to show xticks and xlabel - hide for paper 3
show_xlabel = {'v1_0': False,
               'v1_1': False,
               'v2_0': False,
               'v2_1': False,
               'v2_2': True}

# --- X axis bin edges --- #
# Xticks
xticks = np.array([
    0.4, 0.6, 0.8, 1,
    1.2, 1.4, 1.6, 1.8, 2,
    2.2, 2.4, 2.6, 2.8,
    3, 5, 10, 20, 30])
# Indices of xticks to show
xticks_to_show = np.array(
    [0.4, 1, 2, 3, 10, 30])
last_tick = '>20'
xticks_to_show_ind = [np.where(xticks==i)[0][0]
                      for i in xticks_to_show]
# Construct bins
bin_edgeds = edges_from_centers(xticks)
bins = []
for i in range(len(xticks)):
    bins.append((bin_edgeds[i], bin_edgeds[i+1]))
bins = pd.IntervalIndex.from_tuples(bins)
# Construct bins to xticks mapping
dict_bins_to_xticks = {}  # {bin: xtick}
for i in range(len(xticks)):
    dict_bins_to_xticks[bins[i]] = xticks[i]

# Plot both: SMAP only; SMAP & GLDAS
for plot_gldas in [True, False]:
    for v in ['v1', 'v2']:
        if v == 'v1':
            n_coef = 2
        elif v == 'v2':
            n_coef = 3
            
        for i in range(n_coef):
            # --- Plot --- #
            fig = plt.figure(figsize=(8, 7))
            # Create gridspec structure
            gs = gridspec.GridSpec(
                2, 1, height_ratios=[0.2, 0.8])
            ax2 = plt.subplot(gs[1, 0])
    
            # Extract coef and plot from both GLDAS and SMAP
            dict_df_to_plot = {}
            for data_source in ['SMAP', 'GLDAS']:
                if plot_gldas is False and data_source == 'GLDAS':
                    continue
                key = '{}_{}'.format(data_source, v)
                # Extract coef and convert
                coef_all = dict_list_coef[key][i].values.flatten()
                # Convert coef
                if i == 0:
                    coef_all = - 1 / (coef_all * 24)  # [day]
                elif i == 1:
                    coef_all = coef_all * soil_depth[data_source]
                # Put into df together with aridity
                aridity_all = dict_da_aridity[data_source].values.flatten()
                array = np.stack((aridity_all, coef_all),
                                 axis=1) # [n_pixel, 2]
                df = pd.DataFrame(array, columns=['arid', 'coef'])
                df = df.dropna()
                # Bin to aridity and calculate median and 5th-95th quantile
                ts_arid_bin = pd.cut(df['arid'], bins=bins)
                ts_arid_xticks = \
                    [dict_bins_to_xticks[bin]
                     for bin in ts_arid_bin]
                df['arid'] = ts_arid_xticks
                ts_median = df.groupby('arid').median()
                ts_5th = df.groupby('arid').quantile(q=0.25)
                ts_95th = df.groupby('arid').quantile(q=0.75)
                ts_count = df.groupby('arid').count()  # Count of each bin
                df_to_plot = pd.concat(
                    [ts_median, ts_5th, ts_95th, ts_count],
                    axis=1)
                df_to_plot.columns = ['median', 'low', 'high', 'count']
                # If any bin is empty, insert
                for x in xticks:
                    if x not in df_to_plot.index:
                        df_to_plot = df_to_plot.append(
                            pd.Series({'median': 0,
                                       'low': 0,
                                       'high': 0,
                                       'count': 0},
                                      name=x))
                df_to_plot = df_to_plot.sort_index()
                # Put into dict
                dict_df_to_plot[data_source] = df_to_plot
                # Plot
                ax2.plot(
                    range(len(df_to_plot)),
                    df_to_plot['median'].values,
                    color=color[data_source],
                    label=data_source)
                ax2.fill_between(
                    range(len(df_to_plot)),
                    df_to_plot['low'].values,
                    df_to_plot['high'].values,
                    color=color[data_source], alpha=0.2)
            # --- Make the plot look better --- #
            n_plot = len(dict_df_to_plot['SMAP'])
#            if show_xlabel['{}_{}'.format(v, i)] or plot_gldas is False:  # For paper formatting purpose
            # Set xticks
            ax2.set_xticks(xticks_to_show_ind)
            # Replace the last tick
            ax2.set_xticklabels(
                [i for i in xticks[xticks_to_show_ind][:-1]] + \
                 [last_tick])
            # xlabel
            plt.xlabel('Aridity index (PET/P)', fontsize=20)
#            else:
#                ax2.get_xaxis().set_visible(False)
            # xlim and ylim
            plt.xlim([0, n_plot-1])
            plt.ylim(ylim[i])
            # labels
            plt.ylabel(ylabel[i], fontsize=20)
            ax2.tick_params(labelsize=16)
            
            # --- Add count subplot --- #
            ax1 = plt.subplot(gs[0, 0])
            ax1.bar(range(n_plot),
                    dict_df_to_plot['SMAP']['count'],
                    color='grey', width=0.8)
            plt.xlim([0, n_plot-1])
            ax1.tick_params(labelsize=16)
            ax1.get_xaxis().set_visible(False)
            
            # --- Make plot look better --- #
            fig.subplots_adjust(hspace=0.1)
            
            # Save figure
            fig.savefig(
                os.path.join(
                    output_plot_dir,
                    '{}.X_{}.coef_{}.png'.format(
                        'SMAPvsGLDAS' if plot_gldas is True else 'SMAP',
                        v, i+1)),
                format='png', dpi=150,
                bbox_inches='tight', pad_inches=0)
            
    # --- Get legend --- #
    fig = plt.figure(figsize=(8, 7))
    for data_source in ['SMAP', 'GLDAS']:
        plt.plot(range(20), range(20),
                 color=color[data_source],
                 label=data_source)
    plt.legend(fontsize=16)
    fig.savefig(os.path.join(
        output_plot_dir, 'legend.png'),
    format='png', dpi=150,
    bbox_inches='tight', pad_inches=0)

