
#!/usr/bin/env python3
"""
Plot AGCD precipitation comparison with model experiments

Script to plot AGCD precipitation data alongside model experiment differences
for a single year comparison.

Author: Mathew Lipson <m.lipson@unsw.edu.au>
Date: 2025-09-10
"""

__version__ = "2025-09-10"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

import os
import xarray as xr
import iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xesmf as xe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import glob
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##############################################################################
# Configuration
##############################################################################

# Default dates for comparison
sdate, edate = '2020-02-01', '2020-02-27'

# Paths and experiment configuration
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

def get_agcd(sdate, edate):
    '''
    Get AGCD data between sdate and edate
    sdate, edate : str : 'YYYY-MM-DD'
    return : xarray.Dataset : AGCD data
    '''

    agcdpath = '/g/data/zv2/agcd/v1-0-3/precip/total/r005/01day/'
    agcdpath = '/g/data/zv2/agcd/v1-0-3/precip/calib/r005/01day/'

    # convert sdate and edate to datetime
    sdate_dt, edate_dt = pd.to_datetime(sdate), pd.to_datetime(edate)

    n_days = (edate_dt - sdate_dt).days + 1

    # open files between sdate and edate years
    files = []
    for year in range(sdate_dt.year, edate_dt.year + 1):
        files += glob.glob(f'{agcdpath}/agcd_*_{year}.nc')
    files.sort()

    # open files
    agcd = xr.open_mfdataset(files, combine='by_coords', parallel=True)

    # select time range
    agcd_subset = agcd.sel(time=slice(sdate_dt, edate_dt + pd.Timedelta(days=1)))
    # rename agcd coordinates from lat/lon to latitude/longitude
    agcd_subset = agcd_subset.rename({'lat': 'latitude', 'lon': 'longitude'})

    agcd_mean = agcd_subset['precip'].sum(dim='time')

    # select datetime range to match AGCD using agcd_subste.time_bnds
    agcd_sdate = pd.to_datetime(agcd_subset.time_bnds[0, 0].values)
    agcd_edate = pd.to_datetime(agcd_subset.time_bnds[-1, 1].values)

    return agcd_mean, agcd_sdate, agcd_edate

def load_experiment_data(variable, exp, year, sdate, edate):
    """Load experiment data for a specific year and date range"""
    cylc_id = f'rns_ostia_NA_{year}'
    datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
    filename = f'{datapath}/{variable}/{exp}_{variable}.nc'
    
    if os.path.exists(filename):
        print(f"Loading {filename}")
        ds = xr.open_dataset(filename)
        
        # Get the data variable and convert to mm/hour
        data_vars = [var for var in ds.data_vars if var not in ['time_bnds']]
        data = ds[data_vars[0]] * 3600  # Convert from kg m-2 s-1 to mm/hour
        
        # Select time range to match AGCD
        sdate_dt, edate_dt = pd.to_datetime(sdate), pd.to_datetime(edate)
        data_subset = data.sel(time=slice(sdate_dt, edate_dt))
        
        # Sum over time to get total accumulation
        data_total = data_subset.sum(dim='time')
        
        return data_total
    else:
        print(f"File not found: {filename}")
        return None

def plot_agcd_experiment_comparison(dom, year, sdate, edate):
    """Plot comparison of AGCD with two experiments and their difference"""
    
    print(f"\nCreating AGCD vs experiment comparison for {dom} domain: {sdate} to {edate}")
    
    # Get AGCD data
    agcd_data, agcd_sdate, agcd_edate = get_agcd(sdate, edate)
    
    # Get experiment configuration for this domain
    precip_var = precip_vars[dom]
    exp_list = exps[dom]
    
    print(f"Variable: {precip_var}")
    print(f"Experiments: {exp_list}")
    
    # Load experiment data
    exp_data = {}
    for exp in exp_list:
        data = load_experiment_data(precip_var, exp, year, sdate, edate)
        if data is not None:
            exp_data[exp] = data
        else:
            print(f"Warning: Could not load data for {exp}")

    agcd_regridder = xe.Regridder(agcd_data, data, 'conservative')
    agcd_data = agcd_regridder(agcd_data, keep_attrs=True)

    # Calculate differences (experiments - AGCD)
    control_exp = exp_list[0]
    modified_exp = exp_list[1]
    exp1_diff = exp_data[control_exp] - agcd_data
    exp2_diff = exp_data[modified_exp] - agcd_data
    
    # Set up the plot - 3x2 layout
    # Top row: AGCD, exp1, exp2
    # Bottom row: blank, exp1-AGCD, exp2-AGCD
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})
    
    # Define what goes in each panel
    plot_data = [agcd_data, exp_data[control_exp], exp_data[modified_exp], 
                 None, exp1_diff, exp2_diff]  # None for blank panel
    plot_labels = ['AGCD', control_exp, modified_exp, 
                   '', f'{control_exp} - AGCD', f'{modified_exp} - AGCD']
    
    # Set up consistent color scales
    # Use AGCD max as reference for experiments
    agcd_max = float(agcd_data.max())
    exp_max = max([float(exp_data[exp].max()) for exp in exp_list])
    accum_vmax = np.ceil(max(agcd_max, exp_max) / 50) * 50  # Round up to nearest 50
    accum_vmax = 3000  # Fixed max for accumulation plots
    
    diff_max = max(float(abs(exp1_diff).max()), float(abs(exp2_diff).max()))
    diff_vlim = np.ceil(diff_max / 25) * 25  # Round up to nearest 25
    diff_vlim = 2000 # Fixed max for difference plots
    
    print(f"Using accumulation max: {accum_vmax} mm")
    print(f"Using difference limit: ±{diff_vlim} mm")
    
    for i, (ax, data_to_plot, label) in enumerate(zip(axes.flatten(), plot_data, plot_labels)):
        if i == 3:  # Blank panel (bottom left)
            ax.axis('off')
            continue
            
        if 'AGCD' in label and '-' in label:  # Difference plots
            # Plot difference
            n_levels = 21
            cmap = create_white_center_cmap(n_levels, cmap='RdBu')
            vmin = -diff_vlim
            vmax = diff_vlim
            levels = np.linspace(vmin, vmax, n_levels)
            diff_ticks = levels[::2]  # Every 2nd level
            
            title = f'{label}\nPrecipitation difference'
            cbar_title = f'Precipitation difference [mm]'
            
            im = data_to_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                  levels=levels, extend='both', add_colorbar=False, 
                                  transform=proj)
            
            # Add colorbar for difference plot
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=diff_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            
        else:
            # Plot precipitation accumulation
            data_to_plot = data_to_plot.where(data_to_plot > 0)  # Mask zero values
            
            levels = 21
            accum_levels = np.linspace(0, accum_vmax, levels)
            accum_ticks = accum_levels[::2]  # Every 2nd level
            
            title = f'{label}\nPrecipitation accumulation'
            cbar_title = f'Precipitation [mm]'
            
            im = data_to_plot.plot(ax=ax, cmap='Blues', vmin=0, vmax=accum_vmax,
                                  levels=accum_levels, extend='max', add_colorbar=False,
                                  transform=proj)
            
            # Add colorbar
            cbar = custom_cbar(ax, im, cbar_loc='right', ticks=accum_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        
        ax.set_title(title, fontsize=10)
        
        # Add date range in top left corner
        date_text = f'{sdate} to {edate}'
        ax.text(0.02, 0.98, date_text, transform=ax.transAxes, 
                fontsize=6, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
                zorder=10)
        
        # Add coastlines and set extent
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds_for_cartopy(data_to_plot)
        ax.set_extent([left, right, bottom, top], crs=proj)
        
        # Set labels and ticks
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, labelsize=7)
    
    # Save figure
    fname = f'{plotpath}/agcd_vs_{precip_var}_{dom}_{year}.png'
    print(f'Saving figure to {fname}')
    
    # Create output directory
    os.makedirs(plotpath, exist_ok=True)
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    # Cleanup
    plt.close(fig)
    plt.clf()
    plt.cla()
    
    return fname

if __name__ == "__main__":
    """Main function to create AGCD vs experiment comparison plots"""
    
    # Parse command line arguments for flexibility
    parser = argparse.ArgumentParser(description='Plot AGCD vs experiment comparison')
    parser.add_argument('--year', type=str, default='2020', help='Year to analyze (default: 2020)')
    parser.add_argument('--sdate', type=str, default='2021-02-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--edate', type=str, default='2021-02-27', help='End date (YYYY-MM-DD)')
    parser.add_argument('--domain', type=str, choices=['GAL9', 'RAL3P2', 'both'], default='both', 
                       help='Domain to plot (default: both)')
    
    args = parser.parse_args()
    
    print("Starting AGCD vs experiment comparison plotting...")
    print(f"Year: {args.year}")
    print(f"Date range: {args.sdate} to {args.edate}")
    print(f"Output directory: {plotpath}")
    
    # Determine which domains to process
    domains_to_process = doms if args.domain == 'both' else [args.domain]
    
    # Process each domain
    for dom in domains_to_process:
        print(f"\nProcessing domain: {dom}")

        year = args.year
        sdate = args.sdate
        edate = args.edate

        fname = plot_agcd_experiment_comparison(dom, year, sdate, edate)
        if fname:
            print(f"Completed comparison plot for domain {dom}")
        else:
            print(f"Failed to create plot for domain {dom}")
    
    print("\nAll AGCD comparison plotting completed!")