#!/usr/bin/env python3
"""
Plot moist static energy analysis for all experiments

Script to plot multi-year average moist static energy
for all available experiment periods across both domains.

Author: Mathew Lipson <m.lipson@unsw.edu.au>
Date: 2025-09-11
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
years = ['2014','2015', '2016', '2017', '2018', '2019', '2020']
cylc_ids = [f'rns_ostia_NA_{year}' for year in years]
# Use the first cylc_id for the output path
plotpath = f'/g/data/fy29/mjl561/cylc-run/rns_ostia_NA_all/figures'

xmin, xmax, ymin, ymax = 123.64, 140.34, -9.08, -22.64

# Domains and experiments
doms = ['GAL9', 'RAL3P2']
exps = {
    'RAL3P2': ['CCIv2_RAL3P2', 'CCIv2_RAL3P2_mod'],
    'GAL9': ['CCIv2_GAL9', 'CCIv2_GAL9_mod']
}

# Variable name
variable = 'moist_static_energy_3d'

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
        open_success = False
        datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
        filename = f'{datapath}/{variable}/{exp}_{variable}.nc'
        
        if os.path.exists(filename):
            print(f"Loading {filename}")
            try:
                # Use chunks to avoid memory issues with large files
                ds = xr.open_dataset(filename, chunks={'time': 10})
                
                # Get the data variable 
                data_vars = [var for var in ds.data_vars if var not in ['time_bnds']]
                if not data_vars:
                    print(f"  Warning: No data variables found in {filename}")
                    continue
                    
                data = ds[data_vars[0]]
                
                # Add cylc_id as an attribute for tracking
                data.attrs['cylc_id'] = cylc_id
                datasets.append(data)
                print(f"  Successfully loaded {cylc_id} with shape {data.shape}")
                open_success = True

            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue
        else:
            print(f"File not found: {filename}")

        if not open_success:
            print(f'removing {cylc_id} from cylc_ids')
            cylc_ids.remove(cylc_id)
    
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
    
    print(f"Variable: {variable}")
    print(f"Experiments: {exp_list}")
    
    # Load data for all experiments in this domain
    exp_datasets = {}
    for exp in exp_list:
        print(f"Loading all years for experiment: {exp}")
        year_datasets = load_netcdf_data(variable, exp)
        if year_datasets is not None:
            exp_datasets[exp] = year_datasets
            print(f"  Loaded {exp}: {[ds.sizes['time'] for ds in year_datasets]} time steps per year")
        else:
            print(f"Warning: Could not load data for {exp}")
            continue

    if exp_datasets:
        domain_data = {
            'data': exp_datasets,  # dict of exp -> list of yearly datasets
            'variable': variable,
            'experiments': exp_list
        }
        print(f"Successfully loaded data for {dom}")
        return domain_data
    else:
        print(f"No data found for domain {dom}")
        return None

def calculate_mean_mse(domain_data):
    """Calculate multi-year mean moist static energy for each experiment"""
    
    print(f"\nCalculating multi-year mean moist static energy...")
    
    exp_datasets = domain_data['data']  # dict of exp -> list of yearly datasets
    exp_list = domain_data['experiments']

    print(f"Variable: {variable}")
    print(f"Experiments: {exp_list}")
   
    # Calculate mean for each experiment by averaging over time for each year, then averaging those results
    exp_means = {}
    for exp in exp_list:
        print(f"Calculating mean for experiment: {exp}")
        yearly_means = []
        for i, ds in enumerate(exp_datasets[exp]):
            try:
                print(f"  Processing year {i+1}/{len(exp_datasets[exp])} for {exp}...")
                # Process in chunks to avoid memory issues
                yearly_mean = ds.mean(dim='time')
                # Load into memory and close file connections
                yearly_mean = yearly_mean.load()
                yearly_means.append(yearly_mean)
                # Explicitly close the dataset
                ds.close()
                print(f"    Successfully processed year {i+1}")
            except Exception as e:
                print(f"    Error processing year {i+1}: {e}")
                # Try to close the dataset even if there was an error
                try:
                    ds.close()
                except:
                    pass
                continue
        
        if yearly_means:
            exp_mean = sum(yearly_means) / len(yearly_means)  # Average across years
            exp_means[exp] = exp_mean
            print(f"  Successfully calculated mean for {exp} using {len(yearly_means)} years")
        else:
            print(f"  Warning: No valid data found for {exp}")
            continue

    # Check if we have valid data for both experiments
    if len(exp_means) < 2:
        print(f"Warning: Only found data for {len(exp_means)} experiments, need at least 2")
        if len(exp_means) == 1:
            # Return single experiment data
            exp = list(exp_means.keys())[0]
            single_exp = exp_means[exp].expand_dims('experiment').assign_coords(experiment=[exp])
            print(f"Data shape: {single_exp.shape}")
            print(f"MSE range: {single_exp.min().values:.1f} to {single_exp.max().values:.1f} J/kg")
            return single_exp
        else:
            raise ValueError("No valid experiment data found")

    # Create dataset with experiment dimension
    mean_list = []
    for exp in exp_list:
        if exp in exp_means:
            exp_mean = exp_means[exp].expand_dims('experiment').assign_coords(experiment=[exp])
            mean_list.append(exp_mean)

    # Concatenate along experiment dimension
    mean_mse = xr.concat(mean_list, dim='experiment')

    # Add difference as a new experiment (only if we have 2 experiments)
    if len(mean_list) >= 2:
        print("Calculating difference...")
        available_exps = [exp for exp in exp_list if exp in exp_means]
        diff_data = mean_mse.sel(experiment=available_exps[1]) - mean_mse.sel(experiment=available_exps[0])
        diff_data = diff_data.expand_dims('experiment').assign_coords(experiment=['diff'])
        mean_mse = xr.concat([mean_mse, diff_data], dim='experiment')
    
    print(f"Data shape: {mean_mse.shape}")
    print(f"MSE range: {mean_mse.min().values:.1f} to {mean_mse.max().values:.1f} J/kg")
    
    return mean_mse

def plot_spatial(dom, mean_mse, pressure_level, suffix=''):
    """Plot spatial maps of moist static energy at specified pressure level"""

    print(f"\nPlotting spatial MSE at {pressure_level} hPa for domain: {dom}")
    
    exp_list = list(mean_mse.experiment.values)
    exp_list = [exp for exp in exp_list if exp != 'diff']  # Remove diff for now
    
    # Check if we have difference data
    has_diff = 'diff' in mean_mse.experiment.values
    
    # Select pressure level
    data_to_plot = mean_mse.sel(pressure=pressure_level, method='nearest')
    
    # Set up the plot - adjust number of panels based on available data
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    if has_diff and len(exp_list) >= 2:
        # Full plot with difference
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                                sharex=True, sharey=True,
                                subplot_kw={'projection': proj})
        plot_list = exp_list + ['diff']
    elif len(exp_list) >= 2:
        # Two experiments, no difference
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),
                                sharex=True, sharey=True,
                                subplot_kw={'projection': proj})
        plot_list = exp_list
    else:
        # Single experiment
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6),
                                subplot_kw={'projection': proj})
        axes = [axes]  # Make it iterable
        plot_list = exp_list
    
    # Set up consistent color scales
    mse_min = min([data_to_plot.sel(experiment=exp).min().values for exp in exp_list])
    mse_max = max([data_to_plot.sel(experiment=exp).max().values for exp in exp_list])

    # round to nearest 1000
    mse_min = np.floor(mse_min / 1000) * 1000
    mse_max = np.ceil(mse_max / 1000) * 1000
    
    if has_diff:
        diff_data = data_to_plot.sel(experiment='diff')
        diff_max = abs(diff_data).max().values
        diff_vlim = np.ceil(diff_max / 1000) * 1000  # Round up to nearest 1000
        print(f"Using difference limit: ±{diff_vlim:.0f} J/kg")
    
    print(f"Using MSE range: {mse_min:.0f} to {mse_max:.0f} J/kg")
    
    for ax, plot_exp in zip(axes, plot_list):
        if plot_exp == 'diff' and has_diff:
            # Plot difference
            data_plot = data_to_plot.sel(experiment='diff')
            n_levels = 21
            cmap = create_white_center_cmap(n_levels, cmap='RdBu_r')
            vmin = -diff_vlim
            vmax = diff_vlim
            levels = np.linspace(vmin, vmax, n_levels)
            diff_ticks = levels[::2]  # Every 2nd level
            
            title = f'{exp_list[1]} - {exp_list[0]}\nMSE difference at {pressure_level} hPa'
            cbar_title = f'MSE difference [J/kg]'
            
            im = data_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                               levels=levels, extend='both', add_colorbar=False, 
                               transform=proj)
            
            # Add colorbar for difference plot
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=diff_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            
        else:
            # Plot MSE for individual experiments
            data_plot = data_to_plot.sel(experiment=plot_exp)
            
            levels = 21
            mse_levels = np.linspace(mse_min, mse_max, levels)
            mse_ticks = mse_levels[::2]  # Every 2nd level
            
            title = f'{plot_exp}\nMSE at {pressure_level} hPa'
            cbar_title = f'MSE [J/kg]'
            
            im = data_plot.plot(ax=ax, cmap='viridis', vmin=mse_min, vmax=mse_max,
                               levels=mse_levels, extend='both', add_colorbar=False,
                               transform=proj)
            
            # Add colorbar
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=mse_ticks)
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
        
        # Add RAL3P2 domain outline if plotting GAL9
        if dom == 'GAL9':
            from matplotlib.patches import Rectangle
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                           linewidth=1, linestyle='--', edgecolor='black', 
                           facecolor='none', transform=proj, zorder=6)
            ax.add_patch(rect)
        
        left, bottom, right, top = get_bounds_for_cartopy(data_plot)
        ax.set_extent([left, right, bottom, top], crs=proj)
        
        # Set labels and ticks
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, labelsize=7)
    
    # Save figure
    fname = f'{plotpath}/{variable}_spatial_{dom}_{pressure_level}hPa{suffix}.png'
    print(f'Saving figure to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    return fname
    
def plot_vertical_profiles(dom, mean_mse, suffix=''):
    """Plot vertical profiles of moist static energy and anomaly profile"""
    
    print(f"\nPlotting vertical profiles for domain: {dom}")
    
    exp_list = list(mean_mse.experiment.values)
    exp_list = [exp for exp in exp_list if exp != 'diff']  # Remove diff for profiles
    
    # Calculate spatial mean for profiles
    spatial_mean = mean_mse.mean(dim=['latitude', 'longitude'])
    
    # Create the profile plot
    plt.close('all')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # Colors for experiments
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: Full profiles for all experiments
    for i, exp in enumerate(exp_list):
        exp_data = spatial_mean.sel(experiment=exp)
        axes[0].plot(exp_data.values, exp_data.pressure, 
                    label=exp, color=colors[i], linewidth=2)
    
    # Plot 2: Anomaly profile (modified - control)
    if len(exp_list) >= 2:
        # Assume first experiment is control, second is modified
        control_data = spatial_mean.sel(experiment=exp_list[0])
        modified_data = spatial_mean.sel(experiment=exp_list[1])
        
        # Calculate anomaly (modified - control)
        anomaly = modified_data - control_data
        
        # Plot anomaly profile
        axes[1].plot(anomaly.values, anomaly.pressure, 
                    color='red', linewidth=3, label=f'{exp_list[1]} - {exp_list[0]}')
        
        # Add zero reference line
        axes[1].axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Set title and labels for anomaly plot
        axes[1].set_ylabel('Pressure [hPa]')
        axes[1].set_xlabel('MSE Anomaly [J/kg]')
        axes[1].set_title(f'MSE Anomaly Profile\n{dom} Domain')
        axes[1].invert_yaxis()  # Higher pressure at bottom
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add reference pressure lines
        axes[1].axhline(500, color='black', linestyle='--', alpha=0.5)
        axes[1].axhline(850, color='black', linestyle=':', alpha=0.5)
        
        # Print some statistics about the anomaly
        print(f"  Anomaly statistics for {dom}:")
        print(f"    Maximum positive anomaly: {anomaly.max().values:.1f} J/kg at {anomaly.pressure[anomaly.argmax()].values:.1f} hPa")
        print(f"    Maximum negative anomaly: {anomaly.min().values:.1f} J/kg at {anomaly.pressure[anomaly.argmin()].values:.1f} hPa")
        print(f"    Surface anomaly: {anomaly.isel(pressure=-1).values:.1f} J/kg")
        
    else:
        # If only one experiment, show message
        axes[1].text(0.5, 0.5, 'Need at least 2 experiments\nfor anomaly calculation', 
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1].set_title(f'MSE Anomaly Profile\n{dom} Domain')
    
    # Format profile subplot
    axes[0].set_ylabel('Pressure [hPa]')
    axes[0].set_xlabel('Moist Static Energy [J/kg]')
    axes[0].set_title(f'MSE Vertical Profile\n{dom} Domain')
    axes[0].invert_yaxis()  # Higher pressure at bottom
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add reference lines to profile
    axes[0].axhline(500, color='black', linestyle='--', alpha=0.5)
    axes[0].axhline(850, color='black', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    plt.tight_layout()
    
    # Save figure
    fname = f'{plotpath}/{variable}_vertical_profiles_{dom}{suffix}.png'
    print(f'Saving vertical profiles to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    return fname

##############################################################################

if __name__ == "__main__":
    """Main function to create moist static energy plots for all domains"""
    print("Starting moist static energy plotting...")
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
        
        # Calculate multi-year mean
        mean_mse = calculate_mean_mse(domain_data)
        
        # Plot spatial maps at two pressure levels
        plot_spatial(dom, mean_mse, 500, suffix='')
        print(f"Completed 500 hPa spatial plotting for domain {dom}")
        
        plot_spatial(dom, mean_mse, 850, suffix='')
        print(f"Completed 850 hPa spatial plotting for domain {dom}")
        
        # Plot vertical profiles
        plot_vertical_profiles(dom, mean_mse, suffix='')
        print(f"Completed vertical profile plotting for domain {dom}")
    
    print("\nAll moist static energy plotting completed!")
