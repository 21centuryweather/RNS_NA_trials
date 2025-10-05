#!/usr/bin/env python3
"""
Plot minimum pressure for all experiments

Script to find the minimum pressure in each grid cell across the 3 months
for all available experiment periods across both domains, then plot the 
mean minimum pressure across all years.

Author: Mathew Lipson <m.lipson@unsw.edu.au>
Date: 2025-09-30
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##############################################################################
# Configuration
##############################################################################

# Paths
years = ['2011', '2012', '2013', '2014','2015', '2016', '2017', '2018', '2019', '2020']
cylc_ids = [f'rns_ostia_NA_{year}' for year in years]
# Use the first cylc_id for the output path
plotpath = f'/g/data/fy29/mjl561/cylc-run/rns_ostia_NA_all/figures'

xmin, xmax, ymin, ymax = 123.64, 140.34, -9.08, -22.64

# Domains and experiments
doms = ['GAL9', 'RAL3P2']
doms = ['GAL9']
exps = {
    'RAL3P2': ['CCIv2_RAL3P2', 'CCIv2_RAL3P2_mod'],
    'GAL9': ['CCIv2_GAL9', 'CCIv2_GAL9_mod']
}

# Variable selection - using surface pressure
pressure_var = 'air_pressure_at_sea_level'

##############################################################################
# Helper Functions
##############################################################################

def get_bounds_for_cartopy(ds, y_dim='latitude', x_dim='longitude'):
    """Get geographic bounds from dataset"""
    left = float(ds[x_dim].min())
    right = float(ds[x_dim].max())
    top = float(ds[y_dim].max())
    bottom = float(ds[y_dim].min())

    resolution_y = (top - bottom) / (ds[y_dim].size - 1)
    resolution_x = (right - left) / (ds[x_dim].size - 1)

    top = round(top + resolution_y/2, 6)
    bottom = round(bottom - resolution_y/2, 6)
    right = round(right + resolution_x/2, 6)
    left = round(left - resolution_x/2, 6)

    if resolution_y < 0:
        top, bottom = bottom, top
    if resolution_x < 0:
        left, right = right, left

    return left, bottom, right, top

def custom_cbar(ax, im, cbar_loc='right', ticks=None):
    """Create a custom colorbar"""
    import matplotlib.ticker as mticker

    if cbar_loc == 'right':
        cax = inset_axes(ax,
            width='4%',
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm)
        cbar.ax.yaxis.set_label_position('left')
    elif cbar_loc == 'far_right':
        cax = inset_axes(ax,
            width='4%',
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.25, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm)
        cbar.ax.yaxis.set_label_position('left')
    else:
        # bottom
        cax = inset_axes(ax,
            width='100%',
            height='4%',
            loc='lower left',
            bbox_to_anchor=(0, -0.15, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm, orientation='horizontal')

    # Set custom ticks if provided
    if ticks is not None:
        cbar.set_ticks(ticks)

    cbar.formatter = mticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-6, 6))
    cbar.update_ticks()

    return cbar

def create_white_center_cmap(n_levels=21, cmap='RdBu'):
    """
    Create a custom colormap with white at the center two levels (one negative, one positive around zero)
    """
    new_levels = n_levels+1 # account for extension beyond min and max

    # Get colors from the base cmap
    base_cmap = plt.colormaps[cmap]
    colors = [base_cmap(i / (new_levels - 1)) for i in range(new_levels)]

    # Replace the center two colors with white (one negative, one positive around zero)
    center = n_levels // 2
    colors[center] = (1.0, 1.0, 1.0, 1.0)  # Level just below zero (negative)
    colors[center + 1] = (1.0, 1.0, 1.0, 1.0)  # Level just above zero (positive))

    # Create the custom colormap
    new_cmap = LinearSegmentedColormap.from_list('white_center', colors, N=new_levels)
    return new_cmap

##############################################################################
# Main Functions
##############################################################################

def load_netcdf_data(variable, exp):
    """Load NetCDF data for a given variable and experiment from ALL available cylc_ids"""
    datasets = []
    
    for cylc_id in cylc_ids:
        datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
        filename = f'{datapath}/{variable}/{exp}_{variable}.nc'
        
        if os.path.exists(filename):
            print(f"Loading {filename}")
            ds = xr.open_dataset(filename)
            
            # Get the data variable and convert from Pa to hPa
            data_vars = [var for var in ds.data_vars if var not in ['time_bnds']]
            data = ds[data_vars[0]] / 100  # Convert from Pa to hPa
            
            # Add cylc_id as an attribute for tracking
            data.attrs['cylc_id'] = cylc_id
            datasets.append(data)
        else:
            print(f"File not found: {filename}")
    
    if datasets:
        # Return list of datasets (one per year)
        print(f"Returning {len(datasets)} datasets for {exp}_{variable}")
        return datasets
    else:
        print(f"Warning: No files found for {exp}_{variable}.nc in any cylc_id")
        return None

def load_dom_data(dom):
    """Load NetCDF data for a single domain and all its experiments from ALL cylc_ids"""
    print(f"\nLoading data for domain: {dom}")
    print(f"Searching in cylc_ids: {cylc_ids}")
    
    exp_list = exps[dom]
    
    print(f"Variable: {pressure_var}")
    print(f"Experiments: {exp_list}")
    
    # Load data for all experiments in this domain
    exp_datasets = {}
    for exp in exp_list:
        print(f"Loading all years for experiment: {exp}")
        year_datasets = load_netcdf_data(pressure_var, exp)
        if year_datasets is not None:
            exp_datasets[exp] = year_datasets
            print(f"  Loaded {exp}: {[ds.sizes['time'] for ds in year_datasets]} time steps per year")
        else:
            print(f"Warning: Could not load data for {exp}")
            continue

    if exp_datasets:
        domain_data = {
            'data': exp_datasets,  # dict of exp -> list of yearly datasets
            'variable': pressure_var,
            'experiments': exp_list
        }
        print(f"Successfully loaded data for {dom}")
        return domain_data
    else:
        print(f"No data found for domain {dom}")
        return None

def plot_mean_minimum_pressure(dom, domain_data, suffix):
    """Plot mean minimum pressure for all experiments in a domain"""

    print(f"\nProcessing mean minimum pressure for domain: {dom}")
    
    exp_datasets = domain_data['data']  # dict of exp -> list of yearly datasets
    variable = domain_data['variable']
    exp_list = domain_data['experiments']

    print(f"Variable: {variable}")
    print(f"Experiments: {exp_list}")
   
    # Calculate minimum pressure for each experiment by finding min over time for each year, then averaging those results
    mean_min_list = []
    for exp in exp_list:
        print(f"Calculating mean minimum pressure for experiment: {exp}")
        yearly_mins = [ds.min(dim='time').compute() for ds in exp_datasets[exp]]
        exp_mean_min = sum(yearly_mins) / len(yearly_mins)  # Average across years
        # Add experiment coordinate
        exp_mean_min = exp_mean_min.expand_dims('experiment').assign_coords(experiment=[exp])
        mean_min_list.append(exp_mean_min)

    # find the number of months in the data in first experiment
    n_months = len(exp_datasets[exp_list[0]]) * 3

    # Concatenate along experiment dimension
    mean_min_pressure = xr.concat(mean_min_list, dim='experiment')

    # Add difference as a new experiment
    print("Calculating difference...")
    diff_data = mean_min_pressure.sel(experiment=exp_list[1]) - mean_min_pressure.sel(experiment=exp_list[0])
    diff_data = diff_data.expand_dims('experiment').assign_coords(experiment=['diff'])
    mean_min_pressure = xr.concat([mean_min_pressure, diff_data], dim='experiment')
    
    print(f"Data shape: {mean_min_pressure.shape}")
    print(f"Mean minimum pressure range: {mean_min_pressure.min().values:.1f} to {mean_min_pressure.max().values:.1f} hPa")
    
    # Set up the plot - 3 panels
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})
    
    plot_list = exp_list + ['diff']
    
    # Set up consistent color scales
    pressure_min = min([mean_min_pressure.sel(experiment=exp).min().values for exp in exp_list])
    pressure_max = max([mean_min_pressure.sel(experiment=exp).max().values for exp in exp_list])
    pressure_min = 985 # hardcode
    pressure_max = 1005 # hardcode
    
    diff_data = mean_min_pressure.sel(experiment='diff')
    diff_max = abs(diff_data).max().values
    diff_vlim = np.ceil(diff_max * 10) / 10  # Round up to nearest 0.1 hPa
    diff_vlim = 5 # hardcode 
    
    print(f"Using pressure range: {pressure_min:.1f} to {pressure_max:.1f} hPa")
    print(f"Using difference limit: ±{diff_vlim:.1f} hPa")
    
    for ax, plot_exp in zip(axes, plot_list):
        if plot_exp == 'diff':
            # Plot pressure difference
            data_to_plot = mean_min_pressure.sel(experiment='diff')
            n_levels = 21
            cmap = create_white_center_cmap(n_levels, cmap='RdBu_r')
            vmin = -diff_vlim
            vmax = diff_vlim
            levels = np.linspace(vmin, vmax, n_levels)
            diff_ticks = levels[::2]  # Every 2nd level
            
            title = f'{exp_list[1]} - {exp_list[0]}\nMean minimum pressure difference'
            cbar_title = f'Pressure difference [hPa]\n({n_months} months)'
            
            im = data_to_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                  levels=levels, extend='both', add_colorbar=False, 
                                  transform=proj)
            
            # Add colorbar for difference plot
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=diff_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            
        else:
            # Plot mean minimum pressure for individual experiments
            data_to_plot = mean_min_pressure.sel(experiment=plot_exp)
            
            levels = 21
            pressure_levels = np.linspace(pressure_min, pressure_max, levels)
            pressure_ticks = pressure_levels[::2]  # Every 2nd level
            
            title = f'{plot_exp}\nMean minimum pressure'
            cbar_title = f'Pressure [hPa]\n({n_months} months)'
            
            im = data_to_plot.plot(ax=ax, cmap='viridis_r', vmin=pressure_min, vmax=pressure_max,
                                  levels=pressure_levels, extend='both', add_colorbar=False,
                                  transform=proj)
            
            # Add colorbar
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=pressure_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        
        ax.set_title(title, fontsize=10)
        
        # Add cylc_id list in top left corner
        cylc_text = '\n'.join([cylc_id.split('_')[-1] for cylc_id in cylc_ids])
        ax.text(0.02, 0.98, cylc_text, transform=ax.transAxes, 
                fontsize=6, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
                zorder=10)
        
        # Add coastlines and set extent
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        
        # Add RAL3P2 domain outline if plotting GAL9
        if dom == 'GAL9':
            from matplotlib.patches import Rectangle
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                           linewidth=1, linestyle='--', edgecolor='black', 
                           facecolor='none', transform=proj, zorder=6)
            ax.add_patch(rect)
        
        left, bottom, right, top = get_bounds_for_cartopy(data_to_plot)
        ax.set_extent([left, right, bottom, top], crs=proj)
        
        # Set labels and ticks
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, labelsize=7)
    
    # Save figure
    fname = f'{plotpath}/mean_minimum_{variable}_{dom}{suffix}.png'
    print(f'Saving figure to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    return fname

##############################################################################

if __name__ == "__main__":

    print('load dask')
    from dask.distributed import Client
    n_workers = int(os.environ['PBS_NCPUS'])
    local_directory = os.path.join(os.environ['PBS_JOBFS'], 'dask-worker-space')
    try:
        print(client)
    except Exception:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=1, 
            local_directory = local_directory)

    """Main function to create mean minimum pressure plots for all domains"""
    print("Starting mean minimum pressure plotting...")
    print(f"Searching for data in cylc_ids: {cylc_ids}")
    print(f"Output directory: {plotpath}")

    # Create output directory
    os.makedirs(plotpath, exist_ok=True)
    
    # Process each domain
    for dom in doms:
        print(f"\nProcessing domain: {dom}")
        
        # Load data for domain
        domain_data = load_dom_data(dom)
        
        if domain_data is None:
            print(f"Skipping {dom} - no data found")
            continue
        
        # Plot mean minimum pressure
        plot_mean_minimum_pressure(dom, domain_data, suffix='')
        print(f"Completed mean minimum pressure plotting for domain {dom}")
    
    print("\nAll mean minimum pressure plotting completed!")