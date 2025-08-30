#!/usr/bin/env python3
"""
Plot precipitation accumulation from pre-existing NetCDF files

This script loads previously processed NetCDF files and creates spatial plots
of stratiform or total precipitation depending on the domain.

Author: Mathew Lipson
Date: 2025-08-30
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##############################################################################
# Configuration
##############################################################################

# Paths
oshome = os.getenv('HOME')
cylc_id = 'rns_ostia_NA'
datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
plotpath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/figures'

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
# Helper Functions (adapted from plot_pp.py)
##############################################################################

def make_mp4(fnamein,fnameout,fps=9,quality=26):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatability across OS.
        fnamein (string): The image files to create animation from, with wildcards (*).
        fnameout (string): The output filename (excluding extension)
        fps (float): The frames per second.
        quality (float): quality ranges 0 to 51, 51 being worst.
    '''

    import glob
    import imageio.v2 as imageio

    # collect animation frames
    fnames = sorted(glob.glob(fnamein))
    if len(fnames)==0:
        print('no files found to process, check fnamein')
        return
    img_shp = imageio.imread(fnames[0]).shape
    out_h, out_w = img_shp[0],img_shp[1]

    # resize output to blocksize for maximum capatability between different OS
    macro_block_size=16
    if out_h % macro_block_size > 0:
        out_h += macro_block_size - (out_h % macro_block_size)
    if out_w % macro_block_size > 0:
        out_w += macro_block_size - (out_w % macro_block_size)

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 1 and 51 inclusive"

    # use ffmpeg command to create mp4
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{fnamein}" \
        -vcodec libx264 -crf {quality} -s {out_w}x{out_h} -pix_fmt yuv420p -y {fnameout}.mp4'
    os.system(command)

    return f'completed, see: {fnameout}.mp4'

def get_bounds(ds):
    """
    Make sure that the bounds are in the correct order
    """
    if 'latitude' in ds.coords:
        y_dim = 'latitude'
    elif 'lat' in ds.coords:
        y_dim = 'lat'
    if 'longitude' in ds.coords:
        x_dim = 'longitude'
    elif 'lon' in ds.coords:
        x_dim = 'lon'

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
    """
    Create a custom colorbar
    """
    import matplotlib.ticker as mticker

    if cbar_loc == 'right':
        cax = inset_axes(ax,
            width='4%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm, ticks=ticks)
    elif cbar_loc == 'far_right':
        cax = inset_axes(ax,
            width='4%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.25, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm, ticks=ticks)
    else:
        # cbar_loc == 'bottom'
        cax = inset_axes(ax,
            width='100%',  # % of parent_bbox width
            height='4%',
            loc='lower left',
            bbox_to_anchor=(0, -0.15, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm=im.norm, orientation='horizontal', ticks=ticks)

    # Set scientific notation for colorbar ticks if needed
    cbar.formatter = mticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-6, 6))  # Use scientific notation for small/large numbers
    cbar.update_ticks()

    return cbar

def create_white_center_cmap(n_levels=15, cmap='RdBu'):
    """
    Create a custom colormap with white at the center two levels (one negative, one positive around zero)
    """
    
    # Get colors from the base cmap
    base_cmap = plt.colormaps[cmap]
    colors = [base_cmap(i / (n_levels - 1)) for i in range(n_levels)]
    
    # Replace the center two colors with white (one negative, one positive around zero)
    center = n_levels // 2
    if n_levels % 2 == 1:  # Odd number of levels
        # For odd levels, center is at zero, make the levels on either side white
        colors[center] = (1.0, 1.0, 1.0, 1.0)  # Level just below zero (negative)
        colors[center + 1] = (1.0, 1.0, 1.0, 1.0)  # Level just above zero (positive)
        # Keep the center level (zero) as original color
    else:  # Even number of levels
        colors[center - 1] = (1.0, 1.0, 1.0, 1.0)  # Lower center level (negative)
        colors[center] = (1.0, 1.0, 1.0, 1.0)      # Upper center level (positive)
    
    # Create the custom colormap
    cmap = LinearSegmentedColormap.from_list('white_center', colors, N=n_levels)
    return cmap

def get_variable_opts(variable):
    """
    Get plotting options for precipitation variables
    """
    opts = {
        'plot_title': variable.replace('_', ' '),
        'plot_fname': variable.replace(' ', '_'),
        'units': '?',
        'vmin': None, 
        'vmax': None,
        'cmap': 'viridis',
    }
    
    if variable == 'total_precipitation_rate':
        opts.update({
            'plot_title': 'precipitation rate',
            'plot_fname': 'total_precipitation_rate',
            'units': 'mm/hour',
            'vmin': 0,
            'vmax': 30,  # 0.01 kg m-2 s-1 * 3600 = 30 mm/hour
            'cmap': 'Blues',
        })
                
    elif variable == 'stratiform_rainfall_flux':
        opts.update({
            'plot_title': 'stratiform rainfall flux',
            'plot_fname': 'stratiform_rainfall_flux', 
            'units': 'mm/hour',
            'vmin': 0,
            'vmax': 30,  # 0.01 kg m-2 s-1 * 3600 = 30 mm/hour
            'cmap': 'Blues',
        })

    return opts

##############################################################################
# Main plotting functions
##############################################################################

def load_netcdf_data(variable, dom, exp):
    """
    Load NetCDF data for a given variable, domain, and experiment
    """
    filename = f'{datapath}/{variable}/{exp}_{variable}.nc'
    
    if not os.path.exists(filename):
        print(f"Warning: File not found: {filename}")
        return None
    
    print(f"Loading {filename}")
    ds = xr.open_dataset(filename)
    
    # Return the data variable (assuming it's the only data variable)
    data_vars = [var for var in ds.data_vars if var not in ['time_bnds']]
    data = ds[data_vars[0]]
    # Convert from kg m-2 s-1 to mm/hour
    data = data * 3600
    return data

def plot_precipitation_single_frame(ds, opts, dom, time_index, ds_cumsum, suffix=''):
    """
    Plot a single frame of precipitation data for two experiments and their difference
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Original dataset with both experiments
    opts : dict
        Plotting options
    dom : str
        Domain name
    time_index : int
        Time index to plot
    ds_cumsum : xarray.Dataset
        Pre-calculated cumulative sum dataset including experiments and 'diff'
    suffix : str, optional
        Additional suffix for filename
    """
    # Get experiment list and time information
    exp_list = exps[dom]
    time_str = str(ds.time.values[time_index])[:19].replace('T', ' ')
    
    # Select single time frame for experiments
    ds_frame = ds.isel(time=time_index).compute()
    
    # Get difference data for this timestep and calculate total for consistent colorbar scaling
    diff_data = ds_cumsum.sel(experiment='diff').isel(time=time_index)
    diff_total = ds_cumsum.sel(experiment='diff').isel(time=-1)
    diff_max = abs(diff_total).max().values

    # Set up the plot - always 3 panels (exp1, exp2, diff)
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})
    plot_list = exp_list + ['diff']

    for ax, plot_exp in zip(axes, plot_list):
        if plot_exp == 'diff':
            # Plot difference
            data_to_plot = diff_data
            n_levels = 21
            cmap = create_white_center_cmap(n_levels, cmap='RdBu')  # Custom colormap with white center
            # Round diff_max up to nearest ten for symmetric colorbar
            vlim = np.ceil(diff_max / 10) * 10
            vlim = 500 # hardcode to 500 for now
            vmin = -vlim
            vmax = vlim
            title = f'{exp_list[1]} - {exp_list[0]} cumulative difference\n{time_str}'
            cbar_title = f'cumulative difference {opts["plot_title"]} [{opts["units"]}]'
            # Create symmetric levels around zero
            levels = np.linspace(vmin, vmax, n_levels)
        else:
            # Plot experiment data
            data_to_plot = ds_frame.sel(experiment=plot_exp)
            cmap = opts['cmap']
            vmin = opts['vmin']
            vmax = opts['vmax']
            title = f'{plot_exp} at {time_str}'
            levels = np.linspace(vmin, vmax, 16)
            cbar_title = f'{opts["plot_title"]} [{opts["units"]}]'
            
            # For precipitation plots, mask zero values to show as white
            data_to_plot = data_to_plot.where(data_to_plot > 0)

        im = data_to_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels,
                    extend='both', add_colorbar=False, transform=proj)
                           
        ax.set_title(title, fontsize=10)

        # Add coastlines and set extent
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds(ds_frame)
        ax.set_extent([left, right, bottom, top], crs=proj)

        # Set labels and ticks
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=7) 

        # Add colorbar
        cbar = custom_cbar(ax, im, cbar_loc='bottom')
        cbar.ax.set_xlabel(cbar_title, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Save figure
    time_suffix = f'_t{time_index:05d}' if 'time' in ds.dims else ''
    fname = f'{plotpath}/{opts["plot_fname"]}_single_frame_{dom}{time_suffix}{suffix}.png'
    print(f'Saving figure to {fname}')

    fig.savefig(fname, dpi=300, bbox_inches='tight')

    return fname

def plot_time_mean(ds, opts, dom, suffix=''):
    """
    Plot time-averaged precipitation data
    """
    # Calculate time mean
    if 'time' in ds.dims:
        ds_mean = ds.mean(dim='time')
    else:
        ds_mean = ds
        
    ds_mean = ds_mean.compute()

    # Calculate difference
    exp_list = exps[dom]
    if len(exp_list) == 2:
        diff_data = ds_mean.sel(experiment=exp_list[1]) - ds_mean.sel(experiment=exp_list[0])
        diff_max = abs(diff_data).max().values
        has_diff = True
    else:
        has_diff = False

    # Set up the plot
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    if has_diff:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                                sharex=True, sharey=True,
                                subplot_kw={'projection': proj})
        plot_list = exp_list + ['diff']
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(exp_list), figsize=(6*len(exp_list), 6),
                                sharex=True, sharey=True,
                                subplot_kw={'projection': proj})
        if len(exp_list) == 1:
            axes = [axes]
        plot_list = exp_list

    for ax, plot_exp in zip(axes, plot_list):
        if plot_exp == 'diff' and has_diff:
            # Plot difference
            data_to_plot = diff_data
            n_levels = 21
            cmap = create_white_center_cmap(n_levels)  # Custom colormap with white center
            vmin = -diff_max
            vmax = diff_max
            title = f'{exp_list[1]} - {exp_list[0]} difference (time mean)'
            # Create symmetric levels around zero
            levels = np.linspace(vmin, vmax, n_levels)
            cbar_title = f'difference {opts["plot_title"]} [{opts["units"]}]'
        else:
            # Plot experiment data
            data_to_plot = ds_mean.sel(experiment=plot_exp)
            cmap = opts['cmap']
            vmin = opts['vmin']
            vmax = opts['vmax']
            title = f'{plot_exp} (time mean)'
            cbar_title = f'mean {opts["plot_title"]} [{opts["units"]}]'

        # For precipitation plots, mask zero values to show as white
        if plot_exp != 'diff':
            data_to_plot = data_to_plot.where(data_to_plot > 0)

        levels = np.linspace(vmin, vmax, 21)

        im = data_to_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels,
                    extend='both', add_colorbar=False, transform=proj)
                           
        ax.set_title(title, fontsize=10)

        # Add coastlines and set extent
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds(ds_mean)
        ax.set_extent([left, right, bottom, top], crs=proj)

        # Set labels and ticks
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=7) 

        # Add colorbar
        cbar = custom_cbar(ax, im, cbar_loc='bottom')
        cbar.ax.set_xlabel(cbar_title, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Save figure
    fname = f'{plotpath}/{opts["plot_fname"]}_time_mean_{dom}{suffix}.png'
    print(f'Saving figure to {fname}')

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    return fname

##############################################################################
# Main execution
##############################################################################

def main():
    """
    Main function to load data and create precipitation plots
    """
    print("Starting precipitation accumulation plotting...")

    for dom in doms:
        print(f"\nProcessing domain: {dom}")

        # Get the appropriate precipitation variable for this domain
        precip_var = precip_vars[dom]
        opts = get_variable_opts(precip_var)
        
        print(f"  Variable: {precip_var}")
        
        # Load data for all experiments in this domain
        exp_list = exps[dom]
        datasets = []
        
        for exp in exp_list:
            data = load_netcdf_data(precip_var, dom, exp)
            if data is not None:
                # Add experiment coordinate
                data = data.expand_dims('experiment')
                data = data.assign_coords(experiment=[exp])
                datasets.append(data)
            else:
                print(f"  Warning: Could not load data for {exp}")

        # Combine datasets along experiment dimension
        ds = xr.concat(datasets, dim='experiment')
        
        print(f"  Data shape: {ds.shape}")
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

        # # Create time-mean plot
        # print("  Creating time-mean plot...")
        # plot_time_mean(ds, opts, dom)

        print("  Creating single frame plots...")
        
        # Add difference as a new experiment to the original dataset
        print("  Adding difference experiment...")
        diff_data = ds.sel(experiment=exp_list[1]) - ds.sel(experiment=exp_list[0])
        diff_data = diff_data.expand_dims('experiment').assign_coords(experiment=['diff'])
        ds = xr.concat([ds, diff_data], dim='experiment')
        
        # Pre-calculate cumulative sum for all experiments (including diff)
        print("  Pre-calculating cumulative sums...")
        ds_cumsum = ds.cumsum(dim='time')
        
        n_frames = ds.time.size
        n_frames = 5  # Plot first 10 time steps
        for i in range(n_frames):
            print(f"    Plotting time index {i}...")
            plot_precipitation_single_frame(ds, opts, dom, i, ds_cumsum)
        
        # Create MP4 animation from the frame files
        print("  Creating MP4 animation...")
        frame_pattern = f'{plotpath}/{opts["plot_fname"]}_single_frame_{dom}_t*.png'
        mp4_output = f'{plotpath}/{opts["plot_fname"]}_animation_{dom}'
        make_mp4(frame_pattern, mp4_output, fps=24, quality=23)

        # now delete png files that were used to make the movie
        for i in range(n_frames):
            png_file = f'{plotpath}/{opts["plot_fname"]}_single_frame_{dom}_t{i}.png'
            if os.path.exists(png_file):
                os.remove(png_file)

        print(f"  Completed plotting for domain {dom}")

    print("\nPrecipitation plotting completed!")

if __name__ == "__main__":

    ############################################################################

    # dask configuration

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
    
    # make plotpath if it doesn't exist
    os.makedirs(plotpath, exist_ok=True)

    ############################################################################

    main()
