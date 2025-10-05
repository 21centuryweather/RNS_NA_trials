#!/usr/bin/env python3
"""
Plot moist static energy component analysis for all experiments

Script to plot multi-year average MSE components (sensible heat, latent heat, geopotential)
for all available experiment periods across both domains.

MSE = cp*T + Lv*q + g*z
where:
- cp*T: sensible heat component
- Lv*q: latent heat component  
- g*z: geopotential component

Author: Mathew Lipson <m.lipson@unsw.edu.au>
Date: 2025-10-05
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
years = ['2011','2012','2013','2014','2015', '2016', '2017', '2018', '2019', '2020']
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

# Required variables for MSE components
required_vars = ['air_temperature_3d', 'relative_humidity_3d', 'geopotential_height_3d']

# Physical constants
cp = 1004.0  # J/(kg*K) - specific heat capacity of dry air at constant pressure
Lv = 2.5e6   # J/kg - latent heat of vaporization
g = 9.81     # m/s^2 - gravitational acceleration

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

def custom_cbar(ax, im, cbar_title, cbar_loc='right', ticks=None):
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
        cbar.ax.set_ylabel(cbar_title, fontsize=8)
        
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
        cbar.ax.set_ylabel(cbar_title, fontsize=8)

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
        cbar.ax.set_xlabel(cbar_title, fontsize=8)

    # Set custom ticks if provided
    if ticks is not None:
        cbar.set_ticks(ticks)

    cbar.formatter = mticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-6, 6))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=7)

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
    
    print(f"Required variables: {required_vars}")
    print(f"Experiments: {exp_list}")
    
    # Load data for all experiments and all required variables in this domain
    exp_datasets = {}
    for exp in exp_list:
        print(f"Loading all years and variables for experiment: {exp}")
        
        # Load each required variable
        exp_var_data = {}
        all_vars_loaded = True
        
        for variable in required_vars:
            year_datasets = load_netcdf_data(variable, exp)
            if year_datasets is not None:
                exp_var_data[variable] = year_datasets
                print(f"  Loaded {variable} for {exp}: {[ds.sizes['time'] for ds in year_datasets]} time steps per year")
            else:
                print(f"Warning: Could not load {variable} for {exp}")
                all_vars_loaded = False
                break
        
        if all_vars_loaded:
            exp_datasets[exp] = exp_var_data
        else:
            print(f"Warning: Skipping {exp} - not all required variables available")
            continue

    if exp_datasets:
        domain_data = {
            'data': exp_datasets,  # dict of exp -> dict of variable -> list of yearly datasets
            'variables': required_vars,
            'experiments': exp_list
        }
        print(f"Successfully loaded data for {dom}")
        return domain_data
    else:
        print(f"No data found for domain {dom}")
        return None

def calc_specific_humidity(t3d, rh3d):
    """Calculate specific humidity from temperature and relative humidity using Magnus formula
    
    This uses the same method as in plot_pp.py:calc_moist_static_energy
    
    Parameters:
    -----------
    t3d : xarray.DataArray
        3D air temperature (K or °C) with pressure coordinate
    rh3d : xarray.DataArray
        3D relative humidity (% or fraction) with pressure coordinate
        
    Returns:
    --------
    q : xarray.DataArray
        Specific humidity (kg/kg)
    """
    
    print('  Calculating specific humidity from relative humidity...')
    
    # Ensure temperature is in Kelvin
    if t3d.attrs.get('units', '') == '°C':
        print('    Converting temperature from Celsius to Kelvin...')
        T_K = t3d + 273.15
    else:
        T_K = t3d.copy()
    
    # Convert relative humidity from % to fraction if needed
    if rh3d.max() > 2.0:  # Assume it's in percentage if max > 200%
        print('    Converting relative humidity from % to fraction...')
        RH = rh3d / 100.0
    else:
        RH = rh3d.copy()
    
    # Get pressure levels from the data (assume in hPa)
    pressure = T_K.pressure  # in hPa
    
    # Calculate saturation vapor pressure using Magnus formula (hPa)
    # es = 6.112 * exp(17.67 * (T-273.15) / (T-29.65))
    T_C = T_K - 273.15  # Convert to Celsius for Magnus formula
    es = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    
    # Calculate saturation mixing ratio (kg/kg)
    # qs = 0.622 * es / (p - 0.378 * es)
    qs = 0.622 * es / (pressure - 0.378 * es)
    
    # Calculate specific humidity from relative humidity
    q = RH * qs
    
    # Set attributes
    q.attrs['units'] = 'kg/kg'
    q.attrs['long_name'] = 'specific humidity'
    q.attrs['description'] = 'calculated from relative humidity using Magnus formula'
    
    print(f'    Specific humidity range: {q.min().values:.6f} to {q.max().values:.6f} kg/kg')
    
    return q

def calculate_mse_components(temp_data, rh_data, height_data):
    """Calculate MSE components from temperature, relative humidity, and height data"""
    
    # Calculate specific humidity from relative humidity
    q_data = calc_specific_humidity(temp_data, rh_data)
    
    # Sensible heat component: cp * T
    sensible_heat = cp * temp_data
    
    # Latent heat component: Lv * q
    latent_heat = Lv * q_data
    
    # Geopotential component: g * z
    geopotential = g * height_data
    
    return sensible_heat, latent_heat, geopotential

def calculate_yearly_means_components(domain_data):
    """Calculate yearly mean MSE components for each experiment"""
    
    print(f"\nCalculating yearly mean MSE components...")
    
    exp_datasets = domain_data['data']  # dict of exp -> dict of variable -> list of yearly datasets
    exp_list = domain_data['experiments']
    variables = domain_data['variables']

    print(f"Variables: {variables}")
    print(f"Experiments: {exp_list}")
   
    # Calculate yearly means for each experiment and component
    exp_yearly_components = {}
    for exp in exp_list:
        print(f"Calculating yearly component means for experiment: {exp}")
        
        if exp not in exp_datasets:
            continue
            
        # Get the number of years (should be same for all variables)
        n_years = len(exp_datasets[exp]['air_temperature_3d'])
        yearly_components = {'sensible_heat': [], 'latent_heat': [], 'geopotential': []}
        
        for i in range(n_years):
            try:
                print(f"  Processing year {i+1}/{n_years} for {exp}...")
                
                # Get data for this year for all variables
                temp_year = exp_datasets[exp]['air_temperature_3d'][i]
                rh_year = exp_datasets[exp]['relative_humidity_3d'][i]
                height_year = exp_datasets[exp]['geopotential_height_3d'][i]
                
                # Calculate temporal mean for each variable
                temp_mean = temp_year.mean(dim='time')
                rh_mean = rh_year.mean(dim='time')
                height_mean = height_year.mean(dim='time')
                
                # Calculate MSE components
                sensible_heat, latent_heat, geopotential = calculate_mse_components(
                    temp_mean, rh_mean, height_mean
                )
                
                # Load into memory and close file connections
                sensible_heat = sensible_heat.load()
                latent_heat = latent_heat.load()
                geopotential = geopotential.load()
                
                yearly_components['sensible_heat'].append(sensible_heat)
                yearly_components['latent_heat'].append(latent_heat)
                yearly_components['geopotential'].append(geopotential)
                
                # Close datasets
                temp_year.close()
                rh_year.close()
                height_year.close()
                
                print(f"    Successfully processed year {i+1}")
                
            except Exception as e:
                print(f"    Error processing year {i+1}: {e}")
                # Try to close datasets even if there was an error
                try:
                    temp_year.close()
                    rh_year.close()
                    height_year.close()
                except:
                    pass
                continue
        
        if yearly_components['sensible_heat']:
            exp_yearly_components[exp] = yearly_components
            print(f"  Successfully calculated yearly component means for {exp} using {len(yearly_components['sensible_heat'])} years")
        else:
            print(f"  Warning: No valid component data found for {exp}")
            continue

    # Check if we have valid data
    if len(exp_yearly_components) < 1:
        raise ValueError("No valid experiment data found")
    
    # Return the yearly component means structure
    yearly_components_data = {
        'yearly_components': exp_yearly_components,  # dict of exp -> dict of component -> list of yearly mean datasets
        'components': ['sensible_heat', 'latent_heat', 'geopotential'],
        'experiments': exp_list
    }
    
    print(f"Successfully calculated yearly component means for {len(exp_yearly_components)} experiments")
    
    return yearly_components_data

def aggregate_to_multiyear_mean_components(yearly_components_data):
    """Aggregate yearly component means to multi-year mean with difference experiments"""
    
    exp_yearly_components = yearly_components_data['yearly_components']
    exp_list = yearly_components_data['experiments']
    components = yearly_components_data['components']
    
    # Calculate multi-year mean for each experiment and component
    exp_component_means = {}
    for exp in exp_list:
        if exp in exp_yearly_components and exp_yearly_components[exp]:
            exp_component_means[exp] = {}
            
            for component in components:
                if component in exp_yearly_components[exp] and exp_yearly_components[exp][component]:
                    # Average across years for this experiment and component
                    component_mean = sum(exp_yearly_components[exp][component]) / len(exp_yearly_components[exp][component])
                    exp_component_means[exp][component] = component_mean
    
    # Create datasets with experiment dimension for each component
    component_datasets = {}
    for component in components:
        mean_list = []
        for exp in exp_list:
            if exp in exp_component_means and component in exp_component_means[exp]:
                exp_mean = exp_component_means[exp][component].expand_dims('experiment').assign_coords(experiment=[exp])
                mean_list.append(exp_mean)

        if mean_list:
            # Concatenate along experiment dimension
            component_mean = xr.concat(mean_list, dim='experiment')

            # Add difference as a new experiment (only if we have 2 experiments)
            if len(mean_list) >= 2:
                available_exps = [exp for exp in exp_list if exp in exp_component_means and component in exp_component_means[exp]]
                diff_data = component_mean.sel(experiment=available_exps[1]) - component_mean.sel(experiment=available_exps[0])
                diff_data = diff_data.expand_dims('experiment').assign_coords(experiment=['diff'])
                component_mean = xr.concat([component_mean, diff_data], dim='experiment')
            
            component_datasets[component] = component_mean
    
    print(f"Multi-year component mean data shapes:")
    for component, data in component_datasets.items():
        print(f"  {component}: {data.shape}")
        if len(data.experiment) > 1:
            exp_data = data[0:-1, ...]  # Exclude diff
            print(f"  {component} range: {exp_data.min().values:.0f} to {exp_data.max().values:.0f} J/kg")
    
    return component_datasets

def plot_spatial_components(dom, component_datasets, pressure_level, suffix=''):
    """Plot spatial maps of MSE components at specified pressure level"""

    print(f"\nPlotting spatial MSE components at {pressure_level} hPa for domain: {dom}")
    
    components = list(component_datasets.keys())
    component_titles = {
        'sensible_heat': 'Sensible Heat (cp×T)',
        'latent_heat': 'Latent Heat (Lv×q)', 
        'geopotential': 'Geopotential (g×z)'
    }
    
    # Get experiment info from first component
    first_component = component_datasets[components[0]]
    exp_list = list(first_component.experiment.values)
    exp_list = [exp for exp in exp_list if exp != 'diff']  # Remove diff for now
    
    # Check if we have difference data
    has_diff = 'diff' in first_component.experiment.values
    
    # Select pressure level for all components
    components_to_plot = {}
    for component in components:
        components_to_plot[component] = component_datasets[component].sel(pressure=pressure_level, method='nearest')
    
    # Set up the plot - 3 components × 3 panels (2 experiments + difference)
    proj = ccrs.PlateCarree()
    plt.close('all')
    
    if has_diff and len(exp_list) >= 2:
        # Full plot with difference: 3 rows (components) × 3 cols (exp1, exp2, diff)
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15),
                                sharex=True, sharey=True,
                                subplot_kw={'projection': proj})
        plot_list = exp_list + ['diff']
    elif len(exp_list) >= 2:
        # Two experiments, no difference: 3 rows × 2 cols
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15),
                                sharex=True, sharey=True,
                                subplot_kw={'projection': proj})
        plot_list = exp_list
    else:
        # Single experiment: 3 rows × 1 col
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 15),
                                subplot_kw={'projection': proj})
        axes = axes.reshape(3, 1)  # Make it 2D for consistency
        plot_list = exp_list
    
    # Set up consistent color scales for each component
    component_ranges = {}
    for component in components:
        data_to_plot = components_to_plot[component]
        comp_min = min([data_to_plot.sel(experiment=exp).min().values for exp in exp_list])
        comp_max = max([data_to_plot.sel(experiment=exp).max().values for exp in exp_list])
        
        # Round to appropriate scale based on magnitude
        magnitude = max(abs(comp_min), abs(comp_max))
        if magnitude > 1e5:
            scale = 10000
        elif magnitude > 1e4:
            scale = 1000
        else:
            scale = 100
            
        comp_min = np.floor(comp_min / scale) * scale
        comp_max = np.ceil(comp_max / scale) * scale
        component_ranges[component] = (comp_min, comp_max)
    
    # Plot each component
    for row, component in enumerate(components):
        data_to_plot = components_to_plot[component]
        comp_min, comp_max = component_ranges[component]
        
        if has_diff:
            diff_data = data_to_plot.sel(experiment='diff')
            diff_max = abs(diff_data).max().values
            magnitude = diff_max
            if magnitude > 1e4:
                scale = 1000
            elif magnitude > 1e3:
                scale = 100
            else:
                scale = 10
            diff_vlim = np.ceil(diff_max / scale) * scale
        
        for col, plot_exp in enumerate(plot_list):
            ax = axes[row, col]
            
            if plot_exp == 'diff' and has_diff:
                # Plot difference
                data_plot = data_to_plot.sel(experiment='diff')
                n_levels = 21
                cmap = create_white_center_cmap(n_levels, cmap='RdBu_r')
                vmin = -diff_vlim
                vmax = diff_vlim
                levels = np.linspace(vmin, vmax, n_levels)
                diff_ticks = levels[::2]  # Every 2nd level
                
                title = f'{exp_list[1]} - {exp_list[0]}\n{component_titles[component]} difference at {pressure_level} hPa'
                cbar_title = f'{component_titles[component]} difference [J/kg]'
                
                im = data_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                   levels=levels, extend='both', add_colorbar=False, 
                                   transform=proj)
                
                # Add colorbar for difference plot
                cbar = custom_cbar(ax, im, cbar_title, cbar_loc='right', ticks=diff_ticks)
                
            else:
                # Plot component for individual experiments
                data_plot = data_to_plot.sel(experiment=plot_exp)
                
                levels = 21
                comp_levels = np.linspace(comp_min, comp_max, levels)
                comp_ticks = comp_levels[::2]  # Every 2nd level
                
                title = f'{plot_exp}\n{component_titles[component]} at {pressure_level} hPa'
                cbar_title = f'{component_titles[component]} [J/kg]'
                
                im = data_plot.plot(ax=ax, cmap='viridis', vmin=comp_min, vmax=comp_max,
                                   levels=comp_levels, extend='both', add_colorbar=False,
                                   transform=proj)
                
                # Add colorbar
                cbar = custom_cbar(ax, im, cbar_title, cbar_loc='right', ticks=comp_ticks)
            
            ax.set_title(title, fontsize=10)
            
            # Add cylc_id list in top left corner (only for first row)
            if row == 0:
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
            
            left, bottom, right, top = get_bounds_for_cartopy(data_plot)
            ax.set_extent([left, right, bottom, top], crs=proj)
            
            # Set labels and ticks
            subplotspec = ax.get_subplotspec()
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
            ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=7)
    
    # plt.tight_layout()
    
    # Save figure
    fname = f'{plotpath}/mse_components_spatial_{dom}_{pressure_level}hPa{suffix}.png'
    print(f'Saving figure to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    return fname
    
def plot_vertical_profiles_components(dom, component_datasets, yearly_components_data=None, heights=None, suffix=''):
    """Plot vertical profiles of MSE components and anomaly profiles"""
    
    print(f"\nPlotting vertical component profiles for domain: {dom}")
    
    components = list(component_datasets.keys())
    component_titles = {
        'sensible_heat': 'Sensible Heat (cp×T)',
        'latent_heat': 'Latent Heat (Lv×q)', 
        'geopotential': 'Geopotential (g×z)'
    }
    
    # Get experiment info from first component
    first_component = component_datasets[components[0]]
    exp_list = list(first_component.experiment.values)
    exp_list = [exp for exp in exp_list if exp != 'diff']  # Remove diff for profiles
    
    # Calculate spatial mean for profiles for each component
    component_spatial_means = {}
    for component in components:
        component_spatial_means[component] = component_datasets[component].mean(dim=['latitude', 'longitude'])
    
    # Create the profile plot - 3 components × 2 panels (absolute, anomaly)
    plt.close('all')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    
    # Colors for experiments
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot each component
    for row, component in enumerate(components):
        spatial_mean = component_spatial_means[component]
        
        # Plot 1: Full profiles for all experiments
        for i, exp in enumerate(exp_list):
            exp_data = spatial_mean.sel(experiment=exp)
            axes[row, 0].plot(exp_data.values, exp_data.pressure, 
                        label=exp, color=colors[i], linewidth=2)
        
        # Plot 2: Anomaly profile (modified - control)
        if len(exp_list) >= 2:
            # Assume first experiment is control, second is modified
            control_data = spatial_mean.sel(experiment=exp_list[0])
            modified_data = spatial_mean.sel(experiment=exp_list[1])
            
            # Calculate anomaly (modified - control)
            anomaly = modified_data - control_data
            
            # Plot individual year anomalies as thin red lines (if yearly_components_data is available)
            if yearly_components_data is not None and len(exp_list) >= 2:
                exp_yearly_components = yearly_components_data['yearly_components']
                control_exp = exp_list[0]
                modified_exp = exp_list[1]
                
                # Get individual yearly means for both experiments and this component
                if (control_exp in exp_yearly_components and modified_exp in exp_yearly_components and
                    component in exp_yearly_components[control_exp] and component in exp_yearly_components[modified_exp]):
                    
                    control_yearly_means = exp_yearly_components[control_exp][component]
                    modified_yearly_means = exp_yearly_components[modified_exp][component]
                    
                    # Plot anomaly for each year
                    n_years = min(len(control_yearly_means), len(modified_yearly_means))
                    
                    # Get pressure range from first year for jitter calculation
                    if n_years > 0:
                        first_control = control_yearly_means[0].mean(dim=['latitude', 'longitude'])
                        pressure_range = first_control.pressure.max().values - first_control.pressure.min().values
                        jitter_range = pressure_range * 0.03  # 3% jitter
                    
                    for i in range(n_years):
                        try:
                            # Calculate spatial mean for each year (data is already temporally averaged)
                            control_year_spatial = control_yearly_means[i].mean(dim=['latitude', 'longitude'])
                            modified_year_spatial = modified_yearly_means[i].mean(dim=['latitude', 'longitude'])
                            
                            # Calculate year anomaly
                            year_anomaly = modified_year_spatial - control_year_spatial
                            
                            # Plot thin red line with original pressure (no jitter)
                            axes[row, 1].plot(year_anomaly.values, year_anomaly.pressure, 
                                        color='red', linewidth=1, alpha=0.25)
                            
                            # Add year label at the bottom with jitter to prevent text overlap
                            if i < len(cylc_ids):
                                year = cylc_ids[i].split('_')[-1]  # Extract year from cylc_id
                                surface_pressure = year_anomaly.pressure.max().values  # Highest pressure (bottom)
                                surface_anomaly = year_anomaly.sel(pressure=year_anomaly.pressure.max(), method='nearest').values
                                
                                # Add jitter only to the text label position
                                np.random.seed(i + row)  # Use deterministic seed for reproducibility
                                pressure_jitter = np.random.uniform(-jitter_range, jitter_range)
                                jittered_label_pressure = surface_pressure + pressure_jitter
                                
                                # Add small text label with jittered position
                                axes[row, 1].text(surface_anomaly, jittered_label_pressure, year, 
                                           fontsize=6, ha='center', va='bottom', 
                                           color='red', alpha=0.7)
                            
                        except Exception as e:
                            print(f"    Warning: Could not plot year {i+1} anomaly for {component}: {e}")
                            continue
            
            # Plot mean anomaly profile (thick red line)
            axes[row, 1].plot(anomaly.values, anomaly.pressure, 
                        color='red', linewidth=3, label=f'{exp_list[1]} - {exp_list[0]}')
            
            # Add zero reference line
            axes[row, 1].axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Print some statistics about the anomaly
            print(f"  {component} anomaly statistics for {dom}:")
            print(f"    Maximum positive anomaly: {anomaly.max().values:.1f} J/kg at {anomaly.pressure[anomaly.argmax()].values:.1f} hPa")
            print(f"    Maximum negative anomaly: {anomaly.min().values:.1f} J/kg at {anomaly.pressure[anomaly.argmin()].values:.1f} hPa")
            print(f"    Surface anomaly: {anomaly.isel(pressure=-1).values:.1f} J/kg")
            
        else:
            # If only one experiment, show message
            axes[row, 1].text(0.5, 0.5, 'Need at least 2 experiments\nfor anomaly calculation', 
                        transform=axes[row, 1].transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # Format profile subplots for this component
        for col in range(2):
            ax = axes[row, col]
            ax.set_ylabel('Pressure [hPa]')
            if col == 0:
                ax.set_xlabel(f'{component_titles[component]} [J/kg]')
                ax.set_title(f'{component_titles[component]} Vertical Profile\n{dom} Domain')
            else:
                ax.set_xlabel(f'{component_titles[component]} Anomaly [J/kg]')
                ax.set_title(f'{component_titles[component]} Anomaly Profile\n{dom} Domain')
            
            if len(exp_list) > 0:
                ax.set_yticks(spatial_mean.pressure.values)
            ax.invert_yaxis()  # Higher pressure at bottom
            ax.grid(True, alpha=0.3)
            ax.legend()

    # Add reference pressure lines
    if heights is not None:
        for ax in axes.flatten():
            for h in heights:
                ax.axhline(h, color='black', linestyle='--', alpha=0.5)

    # Add cylc_id list in top left corner
    cylc_text = '\n'.join([cylc_id.split('_')[-1] for cylc_id in cylc_ids])
    axes[0, 0].text(0.02, 0.9, cylc_text, transform=axes[0, 0].transAxes, 
            fontsize=6, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
            zorder=10)
    
    plt.tight_layout()
    
    # Save figure
    fname = f'{plotpath}/mse_components_vertical_profiles_{dom}{suffix}.png'
    print(f'Saving vertical component profiles to {fname}')
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

    """Main function to create MSE component plots for all domains"""
    print("Starting MSE component plotting...")
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
        
        # Calculate yearly component means
        yearly_components_data = calculate_yearly_means_components(domain_data)
        
        # Aggregate to multi-year mean for spatial plots
        component_datasets = aggregate_to_multiyear_mean_components(yearly_components_data)
        
        # Plot spatial maps at three pressure levels
        plot_spatial_components(dom, component_datasets, 500, suffix='')
        print(f"Completed 500 hPa spatial component plotting for domain {dom}")

        plot_spatial_components(dom, component_datasets, 700, suffix='')
        print(f"Completed 700 hPa spatial component plotting for domain {dom}")
        
        plot_spatial_components(dom, component_datasets, 850, suffix='')
        print(f"Completed 850 hPa spatial component plotting for domain {dom}")
        
        # Plot vertical profiles
        plot_vertical_profiles_components(dom, component_datasets, yearly_components_data=yearly_components_data, heights=[500,700,850], suffix='_with_years')
        print(f"Completed vertical component profile plotting for domain {dom}")

        # Plot vertical profiles
        plot_vertical_profiles_components(dom, component_datasets, yearly_components_data=None, heights=[500,700,850], suffix='')
        print(f"Completed vertical component profile plotting for domain {dom}")

        # plot vertical profile for GAL9 with RAL3P2 domain outline
        if dom == 'GAL9':
            # Subset component datasets
            component_datasets_subset = {}
            for component, data in component_datasets.items():
                component_datasets_subset[component] = data.sel(latitude=slice(ymax,ymin),longitude=slice(xmin,xmax)).compute()
            
            # Also subset the yearly components data to match the spatial subset
            yearly_components_subset = {'yearly_components': {}, 'components': yearly_components_data['components'], 'experiments': yearly_components_data['experiments']}
            for exp in yearly_components_data['experiments']:
                if exp in yearly_components_data['yearly_components']:
                    yearly_components_subset['yearly_components'][exp] = {}
                    for component in yearly_components_data['components']:
                        if component in yearly_components_data['yearly_components'][exp]:
                            yearly_components_subset['yearly_components'][exp][component] = [
                                year_data.sel(latitude=slice(ymax,ymin),longitude=slice(xmin,xmax)) 
                                for year_data in yearly_components_data['yearly_components'][exp][component]
                            ]
            
            plot_vertical_profiles_components(dom, component_datasets_subset, yearly_components_data=yearly_components_subset, heights=[500,700,850], suffix='_within_RAL3P2_with_years')
            print(f"Completed vertical component profile plotting for domain {dom} with RAL3P2 outline")

            plot_vertical_profiles_components(dom, component_datasets_subset, yearly_components_data=None, heights=[500,700,850], suffix='_within_RAL3P2')
            print(f"Completed vertical component profile plotting for domain {dom} with RAL3P2 outline")
    
    print("\nAll MSE component plotting completed!")