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

def load_netcdf_data(variable, exp):
    """Load NetCDF data for a given variable and experiment from ALL available cylc_ids"""
    datasets = []
    
    for cylc_id in cylc_ids:
        datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
        filename = f'{datapath}/{variable}/{exp}_{variable}.nc'
        
        if os.path.exists(filename):
            print(f"Loading {filename}")
            ds = xr.open_dataset(filename)
            
            # Get the data variable and convert to mm/hour
            data_vars = [var for var in ds.data_vars if var not in ['time_bnds']]
            data = ds[data_vars[0]] * 3600  # Convert from kg m-2 s-1 to mm/hour
            
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
    
    # Get the appropriate precipitation variable for this domain
    precip_var = precip_vars[dom]
    exp_list = exps[dom]
    
    print(f"Variable: {precip_var}")
    print(f"Experiments: {exp_list}")
    
    # Load data for all experiments in this domain
    exp_datasets = {}
    for exp in exp_list:
        print(f"Loading all years for experiment: {exp}")
        year_datasets = load_netcdf_data(precip_var, exp)
        if year_datasets is not None:
            exp_datasets[exp] = year_datasets
            print(f"  Loaded {exp}: {[ds.sizes['time'] for ds in year_datasets]} time steps per year")
        else:
            print(f"Warning: Could not load data for {exp}")
            continue

    if exp_datasets:
        domain_data = {
            'data': exp_datasets,  # dict of exp -> list of yearly datasets
            'variable': precip_var,
            'experiments': exp_list
        }
        print(f"Successfully loaded data for {dom}")
        return domain_data
    else:
        print(f"No data found for domain {dom}")
        return None

def plot_total_accumulation(dom, domain_data, suffix):
    """Plot total precipitation accumulation for all experiments in a domain"""

    print(f"\nProcessing total accumulation for domain: {dom}")
    
    exp_datasets = domain_data['data']  # dict of exp -> list of yearly datasets
    precip_var = domain_data['variable']
    exp_list = domain_data['experiments']

    print(f"Variable: {precip_var}")
    print(f"Experiments: {exp_list}")
   
    # Calculate total accumulation for each experiment by summing over time for each year, then summing those results
    total_accum_list = []
    for exp in exp_list:
        print(f"Calculating total accumulation for experiment: {exp}")
        yearly_sums = [ds.sum(dim='time').compute() for ds in exp_datasets[exp]]
        exp_total = sum(yearly_sums)
        # Add experiment coordinate
        exp_total = exp_total.expand_dims('experiment').assign_coords(experiment=[exp])
        total_accum_list.append(exp_total)

    # find the number of months in the data in first experiment
    n_months = len(exp_datasets[exp_list[0]]) * 3

    # Concatenate along experiment dimension
    total_accum = xr.concat(total_accum_list, dim='experiment')

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
    accum_vmax = 1000
    
    diff_max = abs(total_accum.sel(experiment='diff')).max().values
    diff_vlim = np.ceil(diff_max / 50) * 50  # Round up to nearest 50
    diff_vlim = 100
    
    print(f"Using accumulation max: {accum_vmax} mm")
    print(f"Using difference limit: ±{diff_vlim} mm")
    
    for ax, plot_exp in zip(axes, plot_list):
        if plot_exp == 'diff':
            # Plot difference
            data_to_plot = total_accum.sel(experiment='diff') / n_months  # Average per month
            n_levels = 21
            cmap = create_white_center_cmap(n_levels, cmap='RdBu')
            # cmap = plt.colormaps['RdBu']
            vmin = -diff_vlim
            vmax = diff_vlim
            levels = np.linspace(vmin, vmax, n_levels)
            diff_ticks = levels[::2]  # Every 2nd level
            
            title = f'{exp_list[1]} - {exp_list[0]}\nAverage monthly accumulation difference'
            cbar_title = f'Average monthly precipitation difference [mm]'
            
            im = data_to_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                  levels=levels, extend='both', add_colorbar=False, 
                                  transform=proj)
            
            # Add colorbar for difference plot
            cbar = custom_cbar(ax, im, cbar_loc='bottom', ticks=diff_ticks)
            cbar.ax.set_xlabel(cbar_title, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            
        else:
            # Plot total accumulation for individual experiments
            data_to_plot = total_accum.sel(experiment=plot_exp) / n_months  # Average per month
            data_to_plot = data_to_plot.where(data_to_plot > 0)  # Mask zero values
            
            levels = 21
            accum_levels = np.linspace(0, accum_vmax, levels)
            accum_ticks = accum_levels[::2]  # Every 2nd level
            
            title = f'{plot_exp}\nAverage monthly precipitation accumulation'
            cbar_title = f'Average monthly precipitation [mm]'
            
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
    fname = f'{plotpath}/{precip_var}_total_accumulation_{dom}{suffix}.png'
    print(f'Saving figure to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    return fname

def plot_all_accumulation_timeseries(dom, domain_data, suffix):
    """Plots a timeseries of cumulative precipitation for a single domain"""
        
    print(f"\nCreating cumulative precipitation timeseries plot for domain: {dom}")
    
    exp_datasets = domain_data['data']  # dict of exp -> list of yearly datasets
    precip_var = domain_data['variable']
    exp_list = domain_data['experiments']

    print(f"Variable: {precip_var}")
    print(f"Experiments: {exp_list}")

    # Calculate spatial mean and cumulative accumulation for each experiment, for each year, then concatenate results
    print("Calculating spatial mean and cumulative accumulation...")
    exp_cumsum = {}
    for exp in exp_list:
        yearly_cumsums = []
        running_total = 0  # Keep track of cumulative total across years
        for ds in exp_datasets[exp]:
            # Calculate spatial mean over domain, then cumulative sum over time
            spatial_mean = ds.mean(dim=['latitude', 'longitude'])
            cumsum = spatial_mean.cumsum(dim='time').compute()
            # Add the running total from previous years to continue accumulation
            cumsum_adjusted = cumsum + running_total
            yearly_cumsums.append(cumsum_adjusted)
            # Update running total with the final value of this year
            running_total = cumsum_adjusted.isel(time=-1).values
        # Concatenate yearly cumsums along time
        exp_cumsum[exp] = xr.concat(yearly_cumsums, dim='time')
    # Create the timeseries plot
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each experiment
    for exp in exp_list:
        exp_data = exp_cumsum[exp]
        exp_data.plot(ax=ax, label=exp, linewidth=2)
    
    # Format the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Precipitation [mm]')
    ax.set_title(f'Cumulative {precip_var.replace("_", " ").title()} - {dom} Domain\nAll Years: {", ".join([id.split("_")[-1] for id in cylc_ids])}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add text box with cylc_ids
    cylc_text = '\n'.join(cylc_ids)
    ax.text(0.02, 0.98, cylc_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
            zorder=10)
    
    # Save the plot
    fname = f'{plotpath}/{precip_var}_cumulative_timeseries_{dom}{suffix}.png'
    print(f'Saving timeseries plot to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Completed timeseries plot for domain {dom}")
    
    return fname

def plot_monthly_overlay_accumulation(dom, domain_data, suffix):
    """Plots cumulative precipitation timeseries overlaid by month for each year"""
    
    print(f"\nCreating monthly overlay cumulative precipitation plot for domain: {dom}")
    
    exp_datasets = domain_data['data']  # dict of exp -> list of yearly datasets
    precip_var = domain_data['variable']
    exp_list = domain_data['experiments']

    print(f"Variable: {precip_var}")
    print(f"Experiments: {exp_list}")

    # Create the timeseries plot
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for different years
    colors = plt.cm.tab10.colors
    
    # for each experiment, calculate anomoly cumsum of the modified from the control
    # Assuming exp_list has control first, then modified
    control_exp = exp_list[0]  # e.g., 'CCIv2_RAL3P2' 
    modified_exp = exp_list[1]  # e.g., 'CCIv2_RAL3P2_mod'
    
    control_datasets = exp_datasets[control_exp]
    modified_datasets = exp_datasets[modified_exp]
    
    # Collect all anomaly values to determine overall min/max
    all_anomaly_values = []
    all_anomaly_series = []  # Store individual year series for averaging
    
    # Plot anomaly for each year
    for year_idx, (control_ds, modified_ds) in enumerate(zip(control_datasets, modified_datasets)):
        # Calculate spatial mean over domain for both experiments
        control_spatial_mean = control_ds.mean(dim=['latitude', 'longitude'])
        modified_spatial_mean = modified_ds.mean(dim=['latitude', 'longitude'])
        
        # Calculate cumulative sum for both
        control_cumsum = control_spatial_mean.cumsum(dim='time').compute()
        modified_cumsum = modified_spatial_mean.cumsum(dim='time').compute()
        
        # Calculate anomaly (modified - control)
        anomaly_cumsum = modified_cumsum - control_cumsum
        
        # Filter for only first 3 months (Dec, Jan, Feb)
        months = anomaly_cumsum.time.dt.month.values
        mask = (months == 12) | (months == 1) | (months == 2)
        anomaly_filtered = anomaly_cumsum[mask]
        
        # Collect values for overall min/max calculation
        all_anomaly_values.extend(anomaly_filtered.values)
        all_anomaly_series.append(anomaly_filtered.values)
        
        # Calculate timestep indices for each timestep, starting from timestep 1
        time_filtered = anomaly_filtered.time
        timestep_indices = list(range(1, len(time_filtered) + 1))

        # Get year from cylc_id attribute
        year_label = control_ds.attrs.get('cylc_id', f'Year_{year_idx+1}')
        if 'cylc_id' in control_ds.attrs:
            year_label = control_ds.attrs['cylc_id'].split('_')[-1] if '_' in control_ds.attrs['cylc_id'] else year_label
        
        # Plot anomaly line for this year
        color = colors[year_idx % len(colors)]
        ax.plot(timestep_indices, anomaly_filtered.values, 
               linewidth=1, color=color, alpha=0.8)
        
        # Add year label next to the last point
        if len(timestep_indices) > 0 and len(anomaly_filtered.values) > 0:
            last_x = timestep_indices[-1]
            last_y = anomaly_filtered.values[-1]
            ax.text(last_x + 5, last_y, year_label, fontsize=8, 
                   verticalalignment='center', color=color, alpha=0.9)

    # Calculate and plot the average of all years
    if all_anomaly_series:
        # Ensure all series have the same length (they should for the same season)
        min_length = min(len(series) for series in all_anomaly_series)
        truncated_series = [series[:min_length] for series in all_anomaly_series]
        
        # Calculate mean across all years
        average_anomaly = np.mean(truncated_series, axis=0)
        average_timesteps = list(range(1, len(average_anomaly) + 1))
        
        # Plot average line as thick black dashed line
        ax.plot(average_timesteps, average_anomaly, 
               linewidth=2, color='black', linestyle='--', alpha=0.8)

    # Find overall min/max and set symmetric y-limits to center the zero line
    overall_min = min(all_anomaly_values)
    overall_max = max(all_anomaly_values)
    y_limit = max(abs(overall_min), abs(overall_max))
    y_limit = y_limit * 1.1  # Add 10% padding
    ax.set_ylim(-y_limit, y_limit)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    
    # Format the plot
    ax.set_xlabel('')
    ax.set_ylabel('Cumulative Precipitation Anomaly [mm]')
    ax.set_title(f'Cumulative {precip_var.replace("_", " ").title()} Years: {", ".join([id.split("_")[-1] for id in cylc_ids])}')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show key timesteps (approximate for 3-hourly data)
    ts_per_day = 24
    key_timesteps = [1, 15*ts_per_day, 31*ts_per_day, 45*ts_per_day, 62*ts_per_day, 75*ts_per_day, 90*ts_per_day]
    ax.set_xticks(key_timesteps)
    ax.set_xticklabels(['Dec 1', 'Dec 15', 'Dec 31', 'Jan 15', 'Jan 31', 'Feb 15', 'Feb 28'])
    ax.tick_params(axis='x')
    
    # Save the plot
    fname = f'{plotpath}/{precip_var}_monthly_overlay_timeseries_{dom}{suffix}.png'
    print(f'Saving monthly overlay plot to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Completed monthly overlay plot for domain {dom}")
    
    return fname

##############################################################################

if __name__ == "__main__":
    """Main function to create total accumulation plots for all domains"""
    print("Starting total precipitation accumulation plotting...")
    print(f"Searching for data in cylc_ids: {cylc_ids}")
    print(f"Output directory: {plotpath}")
    
    # Create output directory
    os.makedirs(plotpath, exist_ok=True)
    
    # Process each domain for total accumulation maps
    for dom in doms:
    
        print(f"\nProcessing domain: {dom}")
    
        # Load data for dom
        domain_data = load_dom_data(dom)

        # create masked_domain_data
        datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_ids[0]}/netcdf'
        exp1 = domain_data['experiments'][0]
        lsm_fname = f'{datapath}/land_sea_mask/{exp1}_land_sea_mask.nc'
        lsm_mask = xr.open_dataset(lsm_fname).isel(time=0).compute()

        masked_domain_data = {
            'data': domain_data['data'].copy(), # will be replaced below  
            'variable': domain_data['variable'],
            'experiments': domain_data['experiments']
        }
        
        # Overwrite data with masked versions using a loop, preserving DataArray structure
        for exp, datasets in domain_data['data'].items():
            masked_domain_data['data'][exp] = [ds.where(lsm_mask.squeeze().to_array() == 1) for ds in datasets]

        # plot_total_accumulation(dom, domain_data, suffix='')
        # print(f"Completed total accumulation plotting for domain {dom}")

        # plot_total_accumulation(dom, masked_domain_data, suffix='_masked')
        # print(f"Completed total accumulation plotting for domain {dom}")
    
        # plot_all_accumulation_timeseries(dom, masked_domain_data, suffix='_masked')
        # print(f"Completed cumulative timeseries plotting for domain {dom}")
    
        plot_monthly_overlay_accumulation(dom, masked_domain_data, suffix='_masked')
        print(f"Completed monthly overlay plotting for domain {dom}")
    
    # Create cumulative timeseries plots using the same loaded data
    
    print("\nAll precipitation plotting completed!")
