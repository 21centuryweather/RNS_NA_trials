#!/usr/bin/env python3
"""
Plot total precipitation accumulation for all experiments

Script to plot the final cumulative precipitation accumulation 
for all available experiment periods across both domains.

Author: Mathew Lipson <m.lipson@unsw.edu.au>
Date: 2025-08-31
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
cylc_ids = ['rns_ostia_NA_2015', 'rns_ostia_NA_2016', 'rns_ostia_NA_2017', 'rns_ostia_NA_2019', 'rns_ostia_NA_2020']
# Use the first cylc_id for the output path
plotpath = f'/g/data/fy29/mjl561/cylc-run/rns_ostia_NA_all/figures'

# Domains and experiments
doms = ['GAL9', 'RAL3P2']
exps = {
    'RAL3P2': ['CCIv2_RAL3P2', 'CCIv2_RAL3P2_mod'],
    'GAL9': ['CCIv2_GAL9', 'CCIv2_GAL9_mod']
}

# Variable selection based on domain
precip_vars = {
    'RAL3P2': 'stratiform_rainfall_flux',
    'GAL9': 'total_precipitation_rate'
}

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

def create_white_center_cmap(n_levels=15, cmap='RdBu'):
    """Create a custom colormap with white at the center for difference plots"""
    base_cmap = plt.colormaps[cmap]
    colors = [base_cmap(i / (n_levels - 1)) for i in range(n_levels)]
    
    # Replace the center color(s) with white
    center = n_levels // 2
    if n_levels % 2 == 1:
        # Odd number of levels: only replace the center level (which represents zero)
        colors[center] = (1.0, 1.0, 1.0, 1.0)
    else:
        # Even number of levels: replace the two center levels
        colors[center - 1] = (1.0, 1.0, 1.0, 1.0)
        colors[center] = (1.0, 1.0, 1.0, 1.0)
    
    cmap = LinearSegmentedColormap.from_list('white_center', colors, N=n_levels)
    return cmap

##############################################################################
# Main Functions
##############################################################################

def load_netcdf_data(variable, exp):
    """Load NetCDF data for a given variable and experiment from any available cylc_id"""
    for cylc_id in cylc_ids:
        datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
        filename = f'{datapath}/{variable}/{exp}_{variable}.nc'
        
        if os.path.exists(filename):
            print(f"Loading {filename}")
            ds = xr.open_dataset(filename)
            
            # Get the data variable and convert to mm/hour
            data_vars = [var for var in ds.data_vars if var not in ['time_bnds']]
            data = ds[data_vars[0]] * 3600  # Convert from kg m-2 s-1 to mm/hour
            
            return data
    
    print(f"Warning: File not found for {exp}_{variable}.nc in any cylc_id")
    return None

def plot_total_accumulation(dom):
    """Plot total precipitation accumulation for all experiments in a domain"""
    print(f"\nProcessing domain: {dom}")
    
    # Get the appropriate precipitation variable for this domain
    precip_var = precip_vars[dom]
    exp_list = exps[dom]
    
    print(f"Variable: {precip_var}")
    print(f"Experiments: {exp_list}")
    
    # Load data for all experiments in this domain
    datasets = []
    for exp in exp_list:
        data = load_netcdf_data(precip_var, exp)
        if data is not None:
            # Add experiment coordinate
            data = data.expand_dims('experiment')
            data = data.assign_coords(experiment=[exp])
            datasets.append(data)
        else:
            print(f"Warning: Could not load data for {exp}")
            return
    
    # Combine datasets along experiment dimension
    ds = xr.concat(datasets, dim='experiment')
    
    # Calculate total accumulation (sum over time)
    print("Calculating total accumulation...")
    total_accum = ds.sum(dim='time').compute()
    
    # Add difference as a new experiment
    print("Calculating difference...")
    diff_data = total_accum.sel(experiment=exp_list[1]) - total_accum.sel(experiment=exp_list[0])
    diff_data = diff_data.expand_dims('experiment').assign_coords(experiment=['diff'])
    total_accum = xr.concat([total_accum, diff_data], dim='experiment')
    
    print(f"Data shape: {total_accum.shape}")
    print(f"Accumulation range: {total_accum.min().values:.1f} to {total_accum.max().values:.1f} mm")
    
    # Set up the plot - 3 panels (exp1, exp2, diff)
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})
    
    plot_list = exp_list + ['diff']
    
    # Set up consistent color scales
    accum_max = max([total_accum.sel(experiment=exp).max().values for exp in exp_list])
    accum_vmax = np.ceil(accum_max / 100) * 100  # Round up to nearest 100
    
    diff_max = abs(total_accum.sel(experiment='diff')).max().values
    diff_vlim = np.ceil(diff_max / 50) * 50  # Round up to nearest 50
    
    print(f"Using accumulation max: {accum_vmax} mm")
    print(f"Using difference limit: ±{diff_vlim} mm")
    
    for ax, plot_exp in zip(axes, plot_list):
        if plot_exp == 'diff':
            # Plot difference
            data_to_plot = total_accum.sel(experiment='diff')
            n_levels = 21
            cmap = create_white_center_cmap(n_levels, cmap='RdBu')
            cmap = plt.colormaps['RdBu']
            vmin = -diff_vlim
            vmax = diff_vlim
            levels = np.linspace(vmin, vmax, n_levels)
            diff_ticks = levels[::2]  # Every 2nd level
            
            title = f'{exp_list[1]} - {exp_list[0]}\nTotal accumulation difference'
            cbar_title = f'Total precipitation difference [mm]'
            
            im = data_to_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                  levels=levels, extend='both', add_colorbar=False, 
                                  transform=proj)
            
            # Add colorbar for difference plot
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=diff_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            
        else:
            # Plot total accumulation for individual experiments
            data_to_plot = total_accum.sel(experiment=plot_exp)
            data_to_plot = data_to_plot.where(data_to_plot > 0)  # Mask zero values
            
            levels = 21
            accum_levels = np.linspace(0, accum_vmax, levels)
            accum_ticks = accum_levels[::2]  # Every 2nd level
            
            title = f'{plot_exp}\nTotal precipitation accumulation'
            cbar_title = f'Total precipitation [mm]'
            
            im = data_to_plot.plot(ax=ax, cmap='Blues', vmin=0, vmax=accum_vmax,
                                  levels=accum_levels, extend='max', add_colorbar=False,
                                  transform=proj)
            
            # Add colorbar
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=accum_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        
        ax.set_title(title, fontsize=10)
        
        # Add cylc_id list in top left corner
        cylc_text = '\n'.join(cylc_ids)
        ax.text(0.02, 0.98, cylc_text, transform=ax.transAxes, 
                fontsize=6, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
                zorder=10)
        
        # Add coastlines and set extent
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds_for_cartopy(ds)
        ax.set_extent([left, right, bottom, top], crs=proj)
        
        # Set labels and ticks
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, labelsize=7)
    
    # Save figure
    fname = f'{plotpath}/{precip_var}_total_accumulation_{dom}.png'
    print(f'Saving figure to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    # Cleanup
    plt.close(fig)
    plt.clf()
    plt.cla()
    
    return fname

def main():
    """Main function to create total accumulation plots for all domains"""
    print("Starting total precipitation accumulation plotting...")
    print(f"Searching for data in cylc_ids: {cylc_ids}")
    print(f"Output directory: {plotpath}")
    
    # Create output directory
    os.makedirs(plotpath, exist_ok=True)
    
    # Process each domain
    for dom in doms:
        try:
            plot_total_accumulation(dom)
            print(f"Completed plotting for domain {dom}")
        except Exception as e:
            print(f"Error processing domain {dom}: {e}")
            continue
    
    print("\nTotal precipitation accumulation plotting completed!")

##############################################################################

if __name__ == "__main__":
    main()
