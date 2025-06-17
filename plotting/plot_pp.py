__version__ = "2025-06-14"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

'''
Create netcdf from um files

GADI ENVIRONMENT
----------------
module use /g/data/hh5/public/modules; module load conda/analysis3
'''

import time
import os
import xarray as xr
import iris
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
import glob
import sys
import warnings
import importlib
warnings.simplefilter(action='ignore', category=FutureWarning)

##############################################################################

oshome=os.getenv('HOME')
cylc_id = 'rns_ostia_NA'
cycle_path = f'/scratch/fy29/mjl561/cylc-run/{cylc_id}/share/cycle'
datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
plotpath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/figures'

variables = ['wind_speed','moisture_convergence','upward_air_velocity_at_300m','upward_air_velocity_at_1000m']

doms = ['GAL9','RAL3P2']

###############################################################################

def get_exp_path(exp, cycle):

    exp_paths = {
        f'CCIv2_GAL9': f'{cycle_path}/{cycle}/NA/0p11/GAL9/um',
        f'CCIv2_GAL9_mod': f'{cycle_path}/{cycle}/NA/0p11/GAL9_mod/um',
        f'CCIv2_RAL3P2': f'{cycle_path}/{cycle}/NA/0p04/RAL3P2/um',
        f'CCIv2_RAL3P2_mod': f'{cycle_path}/{cycle}/NA/0p04/RAL3P2_mod/um',
    }

    return exp_paths[exp]

def plot_rain_diff(ds, coarsen=False):
    '''plots the two experiments and a difference between them'''

    # calculate mean mm per day
    pr_mean = ds.mean(dim='time') * 86400  # convert from kg m-2 s-1 to mm/day

    # pr_mean = pr.sum(dim='time')
    pr_mean['diff'] = pr_mean[exps[1]] - pr_mean[exps[0]]
    pr_mean = pr_mean.compute()

    if coarsen:
        pr_mean = pr_mean.coarsen({'longitude': 10, 'latitude': 10}, boundary='trim').mean()
        suffix = '_coarsened'
    else:
        suffix = ''

    ######

    proj = ccrs.PlateCarree()
    plt.close('all')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})

    for ax, exp in zip(axes.flatten(), exps+['diff']):
        cmap = opts['cmap'] if exp != 'diff' else 'coolwarm'
        vmin = 0 if exp != 'diff' else -10
        vmax = 20 if exp != 'diff' else 10
        title = exp if exp != 'diff' else f'{exps[1]} - {exps[0]} difference'
        extend = 'both' if vmin < 0 else 'max'
        cbar_title =  'daily precipitation [mm]' if exp != 'diff' else 'difference [mm]'
        levels = np.linspace(vmin, vmax, 11)

        im = pr_mean[exp].plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,levels=levels,
                                extend=extend,add_colorbar=False,transform=proj)
                           
        ax.set_title(title)

        # # for cartopy
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds(ds)
        ax.set_extent([left, right, bottom, top], crs=proj)

        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=7) 

        cbar = custom_cbar(ax,im,cbar_loc='bottom')
        cbar.ax.set_xlabel(cbar_title)
        cbar.ax.tick_params(labelsize=7)

    fname = f'{plotpath}/{opts["plot_fname"]}_diff_{dom}{suffix}.png'
    print(f'saving figure to {fname}')

    fig.savefig(fname, dpi=300, bbox_inches='tight')

def plot_spatial(ds, coarsen=False):
    '''plots the two experiments and a difference between them'''

    ds_mean = ds.mean(dim='time')

    ds_mean['diff'] = ds_mean[exps[1]] - ds_mean[exps[0]]
    ds_mean = ds_mean.compute()

    if coarsen:
        ds_mean = ds_mean.coarsen({'longitude': 10, 'latitude': 10}, boundary='trim').mean()
        suffix = '_coarsened'
    else:
        suffix = ''

    ######

    proj = ccrs.PlateCarree()
    plt.close('all')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})

    for ax, exp in zip(axes.flatten(), exps+['diff']):
        cmap = opts['cmap'] if exp != 'diff' else 'coolwarm'
        vmin = opts['vmin'] if exp != 'diff' else -opts['vmax']/2
        vmax = opts['vmax'] if exp != 'diff' else opts['vmax']/2
        title = exp if exp != 'diff' else f'{exps[1]} - {exps[0]} difference'
        cbar_title =  f'mean {opts["plot_title"]} [{opts["units"]}]' if exp != 'diff' else f'difference {opts["plot_title"]} [{opts["units"]}]'
            
        levels = np.linspace(vmin, vmax, 11)

        im = ds_mean[exp].plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels,
                    extend='both',add_colorbar=False, transform=proj)
                           
        ax.set_title(title)

        # # for cartopy
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds(ds)
        ax.set_extent([left, right, bottom, top], crs=proj)

        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=7)
        ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=7) 

        cbar = custom_cbar(ax,im,cbar_loc='bottom')
        cbar.ax.set_xlabel(cbar_title)
        cbar.ax.tick_params(labelsize=7)

    fname = f'{plotpath}/{opts["plot_fname"]}_diff_{dom}{suffix}.png'
    print(f'saving figure to {fname}')

    fig.savefig(fname, dpi=300, bbox_inches='tight')

    return

def get_exp_cycles(variable, exp, opts):
    '''gets all cycles for a given experiment and variable, saves netcdf and returns xarray dataset'''

    # first check if a netcdf file has already been created
    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
    if os.path.exists(ncfname):
        print(f'netcdf file {ncfname} already exists, loading')
        ds = xr.open_dataset(ncfname)
        varname = list(ds.data_vars)[0]  # get the first variable name
        return ds[varname]

    print(f'processing {variable}')

    cycle_list = sorted([x.split('/')[-2] for x in glob.glob(f'{cycle_path}/*/')])
    assert len(cycle_list) > 0, f"no cycles found in {cycle_path}"

    da_list = []
    for i,cycle in enumerate(cycle_list):
        print('========================')
        print(f'getting {exp} {i}: {cycle}\n')

        exp_path = get_exp_path(exp, cycle)

        # check if first exp in exp_path directory exists, if not drop the cycle from list
        if not os.path.exists(exp_path):
            print(f'path {exp_path} does not exist')
            cycle_list.remove(cycle)
            continue

        # check if any of the variables files are in the directory
        if len(glob.glob(f"{exp_path}/{opts['fname']}*")) == 0:
            print(f'no files in {exp_path}')
            cycle_list.remove(cycle)
            continue

        da = get_um_data(exp_path,opts)

        if da is None:
            print(f'WARNING: no data found at {cycle}')
        else:
            da_list.append(da)
        
        # for time invarient variables (land_sea_mask, surface_altitude) only get the first cycle
        if variable in ['land_sea_mask', 'surface_altitude']:
            print('only needs one cycle')
            break

    print('concatenating, adjusting, computing data')
    try: 
        das = xr.concat(da_list, dim='time')
    except ValueError as e:
        print(f'ValueError: {e}')
        print('no data to concatenate, skipping')
        return xr.DataArray()

    save_netcdf(das, exp, opts)

    return das

def get_um_data(exp_path,opts):
    '''gets UM data, converts to xarray and local time'''

    print(f'processing (constraint: {opts["constraint"]})')

    fpath = f"{exp_path}/{opts['fname']}*"
    try:
        cb = iris.load_cube(fpath, constraint=opts['constraint'])
        # fix timestamp/bounds error in accumulations
        if cb.coord('time').bounds is not None:
            print('WARNING: updating time point to right bound')
            cb.coord('time').points = cb.coord('time').bounds[:,1]
        da = xr.DataArray().from_iris(cb)
    except Exception as e:
        print(f'trouble opening {fpath}')
        print(e)
        return None

    # fix time dimension name if needed
    if ('time' not in da.dims) and (variable not in ['land_sea_mask','surface_altitude']):
        print('WARNING: updating time dimension name from dim_0')
        da = da.swap_dims({'dim_0': 'time'})

    da = filter_odd_times(da)

    if opts['constraint'] in [
        'air_temperature', 
        'soil_temperature', 
        'dew_point_temperature', 
        'surface_temperature'
        ]:

        print('converting from K to °C')
        da = da - 273.15
        da.attrs['units'] = '°C'

    if opts['constraint'] in ['stratiform_rainfall_flux_mean']:
        print('converting from mm/s to mm/h')
        da = da * 3600.
        da.attrs['units'] = 'mm/h'

    if opts['constraint'] in ['moisture_content_of_soil_layer']:
        da = da.isel(depth=opts['level'])

    return da

def filter_odd_times(da):
    '''filters out times that are not the most common minute in the time dimension'''

    if da.time.size == 1:
        return da

    minutes = da.time.dt.minute.values
    most_common_bins = np.bincount(minutes)
    most_common_minutes = np.flatnonzero(most_common_bins == np.max(most_common_bins))
    filtered = np.isin(da.time.dt.minute,most_common_minutes)
    filtered_da = da.sel(time=filtered)

    return filtered_da

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
        left,right = right,left

    return left, bottom, right, top

def custom_cbar(ax,im,cbar_loc='right',ticks=None):
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
    cbar.formatter.set_powerlimits((-3, 3))  # Use scientific notation for small/large numbers
    cbar.update_ticks()

    return cbar

def save_netcdf(das, exp, opts):
    '''saves xarray dataset to netcdf with scale_factor and add_offset if data is between 0 and 1
    Arguments:
        das: xarray DataArray to save
        exp: experiment name
        opts: dictionary of options
    '''
    # convert datarray to dataset
    ds = das.to_dataset(name=opts['plot_fname'])
    ds.attrs = das.attrs.copy()  # Preserve global attributes

    # ensure only single variable in dataset
    assert len(ds.data_vars) == 1, f"dataset has {len(ds.data_vars)} variables, expected 1"

    # drop unnecessary dimensions
    if 'forecast_period' in ds.coords:
        ds = ds.drop_vars('forecast_period')
    if 'forecast_reference_time' in ds.coords:
        ds = ds.drop_vars('forecast_reference_time')

    # chunk to optimise save
    if set(['time', 'longitude', 'latitude']).issubset(ds.dims):
        ilon = ds.sizes['longitude']
        ilat = ds.sizes['latitude']
        ds = ds.chunk({'time': 24, 'longitude': ilon, 'latitude': ilat})
    elif set(['longitude', 'latitude']).issubset(ds.dims):
        ilon = ds.sizes['longitude']
        ilat = ds.sizes['latitude']
        ds = ds.chunk({'longitude': ilon, 'latitude': ilat})

    # encoding
    encoding = {
        'time': {'dtype': 'int32'},
        'longitude': {'dtype': 'float32', '_FillValue': -999},
        'latitude': {'dtype': 'float32', '_FillValue': -999},
    }

    # Assume only one variable in ds
    varname = list(ds.data_vars)[0]
    data_min = float(ds[varname].min().compute())
    data_max = float(ds[varname].max().compute())

    # If all data between -1 and 1, use scale_factor and add_offset for int16 storage
    if -1.0 <= data_min and data_max <= 1.0:
        scale_factor = (data_max - data_min) / (np.iinfo(np.int16).max - np.iinfo(np.int16).min)
        if scale_factor == 0:
            scale_factor = 1.0  # avoid division by zero if data is constant
        add_offset = data_min - np.iinfo(np.int16).min * scale_factor
        encoding[varname] = {
            'dtype': 'int16',
            'scale_factor': scale_factor,
            'add_offset': add_offset,
            '_FillValue': -999,
            'zlib': True,
            'shuffle': True
        }
    else:
        encoding[varname] = {
            'dtype': opts['dtype'],
            '_FillValue': np.nan if 'float' in opts['dtype'] else -999,
            'zlib': True,
            'shuffle': True
        }

    # if STASH attributes exist, convert to string
    if 'STASH' in ds[varname].attrs:
        stash = str(ds[varname].attrs['STASH'])
        ds[varname].attrs['STASH'] = stash
    
    # make datapath and subdirectory if they don't exist
    subdir = os.path.join(datapath, opts["plot_fname"])
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    # save to netcdf
    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
    print(f'saving to netcdf: {ncfname}')
    ds.to_netcdf(ncfname, unlimited_dims='time', encoding=encoding)

    return

def calc_moisture_convergence(u,v,q):
    """
    Calculate moisture convergence from zonal and meridional wind components
    and specific humidity.

    Parameters:
    u : xarray.DataArray
        Zonal wind component (m/s).
    v : xarray.DataArray
        Meridional wind component (m/s).
    q : xarray.DataArray
        Specific humidity (kg/kg).
    Returns:
    moisture_convergence : xarray.DataArray
        Moisture convergence (kg/(m^2 s)).
    """

    print('Calculating moisture convergence...')
    
    # Ensure input DataArrays have the same dimensions
    assert u.shape == v.shape == q.shape, "Input DataArrays must have the same dimensions."
    
    # Compute spatial gradients using finite differences
    dq_dx = q.differentiate("longitude", edge_order=2)
    dq_dy = q.differentiate("latitude", edge_order=2)
    du_dx = u.differentiate("longitude", edge_order=2)
    dv_dy = v.differentiate("latitude", edge_order=2)

    # Compute moisture convergence
    moisture_convergence = -(u * dq_dx + v * dq_dy) - q * (du_dx + dv_dy)
    moisture_convergence.attrs['units'] = 'kg/(m^2 s)'

    return moisture_convergence

def calc_moisture_convergence_emma(u,v,qfluxu,qfluxv):

    print('Calculating moisture convergence...')

    # Ensure input DataArrays have the same dimensions
    assert u.shape == v.shape == qfluxu.shape == qfluxv.shape, "Input DataArrays must have the same dimensions."

    r = 6371000
    dx = r * np.cos(u.latitude * np.pi / 180.0) * np.pi / 180.0
    dy = r * np.pi / 180.0

    sfcDiv = ( v.differentiate('latitude') / dy 
             + u.differentiate('longitude') / dx).to_dataset(name='sfcDiv')
    qDiv = ( qfluxv.differentiate('latitude') / dy 
           + qfluxu.differentiate('longitude') / dx).to_dataset(name='qDiv')

    # units
    # convert to mm/day
    qDiv = qDiv * 86400  # kg/(m^2 s) to mm/day
    qDiv['qDiv'].attrs['units'] = 'mm/day'

    return sfcDiv, qDiv

def get_variable_opts(variable):
    '''standard variable options for plotting. to be updated within master script as needed
    
    constraint: iris constraint for the variable used to extract data from the cube
                this can be a long_name, STASH code, or an iris.Constraint() object
    stash: STASH code for the variable (if available)
    plot_title: title of the plot (spaces allowed)
    plot_fname: description used for filename (spaces not allowed)
    units: units of the variable
    obs_key: key used to describe the obs data (if available)
    obs_period: period to resample the obs data to
    fname: filename of the data file to extract from
    vmin: minimum value for the colorbar for variable
    vmax: maximum value for the colorbar for variable
    cmap: colormap to use for the variable
    threshold: threshold error statitics/ benchmarks (if defined)
    fmt: format string for the variable error statistics
    dtype: data type for saving the variable to netcdf
    level: level of the variable (e.g. soil level, 0-indexed)
    '''

    # standard ops
    opts = {
        'constraint': variable,
        'plot_title': variable.replace('_',' '),
        'plot_fname': variable.replace(' ','_'),
        'units'     : '?',
        'obs_key'   : 'None',
        'obs_period': '1H',
        'fname'     : 'umnsaa_pvera',
        'vmin'      : None, 
        'vmax'      : None,
        'cmap'      : 'viridis',
        'threshold' : None,
        'fmt'       : '{:.2f}',
        'dtype'     : 'float32'
        }
    
    if variable == 'air_temperature':
        opts.update({
            'constraint': 'air_temperature',
            'plot_title': 'air temperature (1.5 m)',
            'plot_fname': 'air_temperature_1p5m',
            'units'     : '°C',
            'obs_key'   : 'Tair',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'threshold' : 2,
            'fmt'       : '{:.2f}',
            })
        
    if variable == 'upward_air_velocity':
        opts.update({
            'constraint': 'upward_air_velocity',
            'plot_title': 'upward air velocity',
            'plot_fname': 'upward_air_velocity',
            'units'     : 'm s-1',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : -1,
            'vmax'      : 1,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })
        
    if variable == 'surface_altitude':
        opts.update({
            'constraint': 'surface_altitude',
            'units'     : 'm',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 2000,
            'cmap'      : 'twilight',
            'dtype'     : 'int16',
            'fmt'       : '{:.0f}',
            })

    elif variable == 'specific_humidity':
        opts.update({
            'constraint': 'm01s03i237',
            'plot_title': 'specific humidity (1.5 m)',
            'plot_fname': 'specific_humidity_1p5m',
            'units'     : 'kg/kg',
            'obs_key'   : 'Qair',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : 0.004,
            'vmax'      : 0.020,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.4f}',
            })

    elif variable == 'latent_heat_flux':
        opts.update({
            'constraint': 'surface_upward_latent_heat_flux',
            'plot_title': 'Latent heat flux',
            'plot_fname': 'latent_heat_flux',
            'units'     : 'W/m2',
            'obs_key'   : 'Qle',
            # 'fname'     : 'umnsaa_pvera',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -100, 
            'vmax'      : 500,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'sensible_heat_flux':
        opts.update({
            'constraint': 'surface_upward_sensible_heat_flux',
            'plot_title': 'Sensible heat flux',
            'plot_fname': 'sensible_heat_flux',
            'units'     : 'W/m2',
            'obs_key'   : 'Qh',
            # 'fname'     : 'umnsaa_pvera',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -100, 
            'vmax'      : 600,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_air_pressure':
        opts.update({
            'units'     : 'Pa',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 88000,
            'vmax'      : 104000,
            'cmap'      : 'viridis',
            })

    elif variable == 'wind_speed_of_gust':
        opts.update({
            'constraint': 'wind_speed_of_gust',
            'plot_title': 'wind speed of gust',
            'plot_fname': 'wind_gust',
            'units'     : 'm/s',
            'obs_key'   : 'Wind_gust',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 10,
            'vmax'      : 40,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'wind_u':
        opts.update({
            'constraint': 'm01s03i225',
            'plot_title': '10 m wind: U-component',
            'plot_fname': 'wind_u_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })

    elif variable == 'wind_v':
        opts.update({
            'constraint': 'm01s03i226',
            'plot_title': '10 m wind: V-component',
            'plot_fname': 'wind_v_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })
    
    elif variable == 'wind_speed':
        opts.update({
            'plot_title': '10 m wind speed',
            'plot_fname': 'wind_speed_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'vmin'      : 0,
            'vmax'      : 10,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'air_pressure_at_sea_level':
        opts.update({
            'constraint': 'air_pressure_at_sea_level',
            'plot_title': 'air pressure at sea level',
            'plot_fname': 'air_pressure_at_sea_level',
            'units'     : 'Pa',
            'obs_key'   : 'SLP',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 97000,
            'vmax'      : 103000,
            'cmap'      : 'viridis',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'total_precipitation_rate':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_flux',
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'precipitation rate',
            'plot_fname': 'total_precipitation_rate',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.5f}',
            })
                
    elif variable == 'convective_rainfall_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s05i201', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'convective rainfall amount',
            'plot_fname': 'convective_rainfall_amount',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'convective_rainfall_flux':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s05i205', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'convective rainfall flux',
            'plot_fname': 'convective_rainfall_flux',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.5f}',
            })
        
    elif variable == 'stratiform_rainfall_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'stratiform_rainfall_flux':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_flux', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2 s-1',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.5f}',
            })
        
    elif variable == 'moisture_flux_u':
        opts.update({
            'constraint': 'm01s30i462',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : None,
            'vmax'      : 0.0002,
            'cmap'      : 'cividis',
            'fmt'       : '{:.6f}',
            })
    
    elif variable == 'moisture_flux_v':
        opts.update({
            'constraint': 'm01s30i463',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : None,
            'vmax'      : 0.0002,
            'cmap'      : 'cividis',
            'fmt'       : '{:.6f}',
            })

    elif variable == 'upward_air_velocity_at_300m':
        opts.update({
            'constraint': iris.Constraint(name='upward_air_velocity', height=300.),
            'units'     : 'm s-1',
            'fname'     : 'umnsaa_pb',
            'vmin'      : -0.04,
            'vmax'      : 0.04,
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'upward_air_velocity_at_1000m':
        opts.update({
            'constraint': iris.Constraint(name='upward_air_velocity', height=1000.),
            'units'     : 'm s-1',
            'fname'     : 'umnsaa_pb',
            'vmin'      : -0.04,
            'vmax'      : 0.04,
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })


    elif variable == 'moisture_convergence':
        opts.update({
            'vmin'      : -20,
            'vmax'      : 20,
            'units'     : 'mm/day',
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })

    # add variable to opts
    opts.update({'variable':variable})

    return opts

if __name__ == "__main__":

    print('running variables:',variables)

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

    for dom in doms:

        ###############################################################################
        
        if dom == 'RAL3P2':
            exps = ['CCIv2_RAL3P2','CCIv2_RAL3P2_mod']
            variables = variables + ['stratiform_rainfall_flux']
            if 'total_precipitation_rate' in variables:
                variables.remove('total_precipitation_rate')
        if dom ==  'GAL9':
            exps = ['CCIv2_GAL9','CCIv2_GAL9_mod']
            variables = variables + ['total_precipitation_rate']
            if 'stratiform_rainfall_flux' in variables:
                variables.remove('stratiform_rainfall_flux')

        ###############################################################################
    
        for variable in variables:
            print(f'processing variable: {variable}')
            opts = get_variable_opts(variable)
            ds = xr.Dataset()
            for exp in exps:
                print(f'processing {exp} {variable}')
                if variable == 'moisture_convergence_old':
                    print('special case for moisture convergence')

                    # check if netcdf file already exists
                    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
                    if os.path.exists(ncfname):
                        print(f'netcdf file {ncfname} already exists, loading')
                        ds[exp] = xr.open_dataset(ncfname)[variable]
                        continue

                    u_opts = get_variable_opts('wind_u')
                    v_opts = get_variable_opts('wind_v')
                    q_opts = get_variable_opts('specific_humidity')

                    u = get_exp_cycles('wind_u', exp, u_opts)
                    v = get_exp_cycles('wind_v', exp, v_opts)
                    q = get_exp_cycles('specific_humidity', exp, q_opts)

                    u_interp = u.interp(latitude=q.latitude, longitude=q.longitude, method='linear')
                    v_interp = v.interp(latitude=q.latitude, longitude=q.longitude, method='linear')

                    ds[exp] = calc_moisture_convergence(u_interp, v_interp, q)
                    save_netcdf(ds[exp], exp, opts)

                elif variable == 'moisture_convergence':
                    print('special case for moisture convergence emma')

                    # check if netcdf file already exists
                    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
                    if os.path.exists(ncfname):
                        print(f'netcdf file {ncfname} already exists, loading')
                        ds[exp] = xr.open_dataset(ncfname)[variable]
                        continue

                    u_opts = get_variable_opts('wind_u')
                    v_opts = get_variable_opts('wind_v')
                    qfluxu_opts = get_variable_opts('moisture_flux_u')
                    qfluxv_opts = get_variable_opts('moisture_flux_v')

                    u = get_exp_cycles('wind_u', exp, u_opts)
                    v = get_exp_cycles('wind_v', exp, v_opts)

                    qfluxu = get_exp_cycles('moisture_flux_u', exp, qfluxu_opts)
                    qfluxv = get_exp_cycles('moisture_flux_v', exp, qfluxv_opts)

                    u_interp = u.interp(latitude=qfluxu.latitude, longitude=qfluxu.longitude, method='linear')
                    v_interp = v.interp(latitude=qfluxv.latitude, longitude=qfluxv.longitude, method='linear')

                    sfcDiv, qDiv = calc_moisture_convergence_emma(u_interp, v_interp, qfluxu, qfluxv)

                    ds[exp] = qDiv['qDiv']
                    save_netcdf(ds[exp], exp, opts)
                elif variable == 'wind_speed':
                    print('special case for wind')

                    u_opts = get_variable_opts('wind_u')
                    v_opts = get_variable_opts('wind_v')
                    u = get_exp_cycles('wind_u', exp, u_opts)
                    v = get_exp_cycles('wind_v', exp, v_opts)
                    ds[exp] = np.sqrt(u**2 + v**2)
                    ds[exp].attrs['units'] = 'm/s'
                    save_netcdf(ds[exp], exp, opts)

                else:
                    # general case for other variables
                    ds[exp] = get_exp_cycles(variable, exp, opts)

            if variable in ['stratiform_rainfall_flux', 'total_precipitation_rate']:
                plot_rain_diff(ds, coarsen=False)
                plot_rain_diff(ds, coarsen=True)

            else:
                plot_spatial(ds, coarsen=False)
                plot_spatial(ds, coarsen=True)

