__version__ = "2025-06-14"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

'''
Create netcdf from um files

GADI ENVIRONMENT
----------------
module use /g/data/hh5/public/modules; module load conda/analysis3
'''

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
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##############################################################################

oshome=os.getenv('HOME')
overwrite_nc = False

variables = ['wind_speed','moisture_convergence','upward_air_velocity_at_300m','upward_air_velocity_at_1000m']
variables = ['total_precipitation_rate']
variables = ['moisture_convergence']
variables = ['low_cloud_amount_instant']
variables = ['medium_cloud_amount_instant']
variables = ['high_cloud_amount_instant']
variables = ['wind_quiver']
variables = ['surface_roughness_length']
variables = ['boundary_layer_depth']
variables = ['convective_rainfall_flux']
variables = ['precip_w_cross_correlation']
variables = ['wet_bulb_potential_temperature_3d']
variables = ['moisture_convergence_old']
variables = ['latent_heat_flux']
variables = ['geopotential_height_3d']
variables = ['relative_humidity_3d']
variables = ['air_temperature_3d']
variables = ['moist_static_energy_3d']
variables = ['upward_air_velocity_3d']
variables = ['wind_speed']
variables = ['stratiform_rainfall_flux','total_precipitation_rate']
variables = ['air_pressure_at_sea_level']
variables = ['moisture_convergence_emma']

doms = ['RAL3P2']
doms = ['GAL9','RAL3P2']
doms = ['GAL9']

###############################################################################

def main(datapath, plotpath, cycle_path, doms, cylc_id):

    for dom in doms:

        print('processing domain:', dom)

        ###############################################################################
        
        if dom == 'RAL3P2':
            exps = ['CCIv2_RAL3P2','CCIv2_RAL3P2_mod']
        if dom ==  'GAL9':
            exps = ['CCIv2_GAL9','CCIv2_GAL9_mod']

        ###############################################################################
    
        for variable in variables:
            if ((dom == 'RAL3P2') and (variable == 'total_precipitation_rate') or
                    (dom == 'GAL9') and (variable == 'stratiform_rainfall_flux')):
                print('skipping')
                continue
                
            print(f'processing variable: {variable}')
            opts = get_variable_opts(variable)
            
            # Determine required variables for this computation
            if variable == 'moisture_convergence_old':
                required_vars = ['wind_u', 'wind_v', 'specific_humidity']
            elif variable == 'moisture_convergence_emma':
                required_vars = ['wind_u', 'wind_v', 'moisture_flux_u', 'moisture_flux_v']
            elif variable == 'moist_static_energy_3d':
                required_vars = ['air_temperature_3d', 'relative_humidity_3d', 'geopotential_height_3d']
            elif variable == 'wind_speed':
                required_vars = ['wind_u', 'wind_v']
            elif variable == 'wind_quiver':
                required_vars = ['wind_u', 'wind_v']
            elif variable == 'precip_w_cross_correlation':
                required_vars = ['stratiform_rainfall_flux', 'upward_air_velocity_3d']
            else:
                required_vars = [variable]
            
            print(f'Required variables: {required_vars}')
            
            # PRE-LOAD ALL REQUIRED DATA FOR THIS VARIABLE
            cached_data = {}
            for exp in exps:
                cached_data[exp] = {}
                for req_var in required_vars:
                    req_opts = get_variable_opts(req_var)
                    cached_data[exp][req_var] = get_cached_data(req_var, exp, req_opts, datapath, cycle_path)
            
            print('All required data loaded/cached')

            ds = xr.Dataset()
            ################################
            if variable == 'wind_quiver':
                u = xr.Dataset()
                v = xr.Dataset()
                
            for exp in exps:
                print(f'processing {exp} {variable} {cylc_id}')
                
                if variable == 'moisture_convergence_old':
                    print('special case for moisture convergence')

                    # check if netcdf file already exists
                    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
                    if os.path.exists(ncfname) and not overwrite_nc:
                        print(f'netcdf file {ncfname} already exists, loading')
                        ds[exp] = xr.open_dataset(ncfname)[variable]
                        continue

                    # Use cached data instead of reloading
                    u = cached_data[exp]['wind_u']
                    v = cached_data[exp]['wind_v']
                    q = cached_data[exp]['specific_humidity']

                    u_interp = u.interp(latitude=q.latitude, longitude=q.longitude, method='linear')
                    v_interp = v.interp(latitude=q.latitude, longitude=q.longitude, method='linear')

                    ds[exp] = calc_moisture_convergence(u_interp, v_interp, q)
                    save_netcdf(ds[exp], exp, opts)

                elif variable == 'moisture_convergence_emma':
                    print('special case for moisture convergence emma')

                    # check if netcdf file already exists
                    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
                    if os.path.exists(ncfname) and not overwrite_nc:
                        print(f'netcdf file {ncfname} already exists, loading')
                        ds[exp] = xr.open_dataset(ncfname)[variable]
                    else:
                        # Use cached data instead of reloading
                        u = cached_data[exp]['wind_u']
                        v = cached_data[exp]['wind_v']
                        qfluxu = cached_data[exp]['moisture_flux_u']
                        qfluxv = cached_data[exp]['moisture_flux_v']

                        u_interp = u.interp(latitude=qfluxu.latitude, longitude=qfluxu.longitude, method='linear')
                        v_interp = v.interp(latitude=qfluxv.latitude, longitude=qfluxv.longitude, method='linear')

                        qDiv = calc_moisture_convergence_emma(u_interp, v_interp, qfluxu, qfluxv)

                        ds[exp] = qDiv['qDiv']
                        save_netcdf(ds[exp], exp, opts)

                elif variable == 'moist_static_energy_3d':
                    print('special case for moist static energy 3d')

                    # check if netcdf file already exists
                    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
                    if os.path.exists(ncfname) and not overwrite_nc:
                        print(f'netcdf file {ncfname} already exists, loading')
                        ds[exp] = xr.open_dataset(ncfname)[variable]
                    else:
                        # Use cached data instead of reloading
                        t3d = cached_data[exp]['air_temperature_3d']
                        rh3d = cached_data[exp]['relative_humidity_3d']
                        z3d = cached_data[exp]['geopotential_height_3d']

                        ds[exp] = calc_moist_static_energy(t3d, rh3d, z3d)
                        save_netcdf(ds[exp], exp, opts)

                elif variable == 'wind_speed':
                    print('special case for wind speed')

                    # Use cached data instead of reloading
                    u = cached_data[exp]['wind_u']
                    v = cached_data[exp]['wind_v']
                    ds[exp] = np.sqrt(u**2 + v**2)
                    ds[exp].attrs['units'] = 'm/s'
                    save_netcdf(ds[exp], exp, opts)

                elif variable == 'wind_quiver':
                    print('special case for wind quiver')
                    
                    # Use cached data instead of reloading
                    u[exp] = cached_data[exp]['wind_u']
                    v[exp] = cached_data[exp]['wind_v']

                elif variable == 'precip_w_cross_correlation':
                    print('special case for precipitation-vertical wind cross-correlation')
                    
                    # Use cached data instead of reloading
                    precip = cached_data[exp]['stratiform_rainfall_flux']
                    w_3d = cached_data[exp]['upward_air_velocity_3d']
                    
                    # Get vertical wind at level
                    level = 850
                    w_height = w_3d.sel(pressure=level)

                    # Calculate cross-correlation
                    ds[exp] = calc_cross_correlation(precip, w_height)

                else:
                    # general case for other variables - use cached data
                    ds[exp] = cached_data[exp][variable]

            # ds_500 = ds.sel(pressure=500)
            # opts_500 = opts.copy()
            # opts_500.update({
            #     'plot_title': 'upward air velocity over land at 500hPa',
            #     'plot_fname': 'upward_air_velocity_over_land_at_500hPa',
            #     'vmin': -0.02,
            #     'vmax': 0.02

            # })
            # plot_spatial(ds_500, opts_500, coarsen=False)
            # plot_spatial(ds_500, opts_500, coarsen=True)

            # # plot_vertical_velocity_profiles(ds.sel(time=slice('2020-01-05','2020-02-05')), exps, suffix='_jan')


            if variable in ['stratiform_rainfall_flux', 'total_precipitation_rate']:
                if ((dom == 'RAL3P2') and (variable == 'stratiform_rainfall_flux') or
                    (dom == 'GAL9') and (variable == 'total_precipitation_rate')):
                    ds = ds.compute()
                    plot_rain_diff(ds, exps, opts, plotpath, dom, coarsen=False)
                    plot_rain_diff(ds, exps, opts, plotpath, dom, coarsen=True, suffix='_coarsened')
                    plot_cumsum(ds, opts, exps, datapath, cycle_path, plotpath, dom)

                    # plot_hours = None
                    # plot_hours = [0,3,6,9,12,15,18,21]
                    # if plot_hours is not None:
                    #     for hour in plot_hours:
                    #         plot_rain_diff(ds.sel(time=ds.time.dt.hour==hour), coarsen=False, suffix=f'_hour_{hour}')
                    #         plot_rain_diff(ds.sel(time=ds.time.dt.hour==hour), coarsen=True, suffix=f'_coarsened_hour_{hour}')
                    
            elif variable == 'wind_quiver':
                print('plotting wind quiver')
                u,v = u.compute(),v.compute()
                plot_wind_quiver(u, v, suffix='_full_period')

                plot_months = None
                plot_months = [12,1,2]
                if plot_months is not None:
                    for month in plot_months:
                        plot_wind_quiver(u.sel(time=u.time.dt.month==month), v.sel(time=v.time.dt.month==month), suffix=f'_month_{month}')

                plot_hours = None
                plot_hours = [0,3,6,9,12,15,18,21]
                if plot_hours is not None:
                    for hour in plot_hours:
                        plot_wind_quiver(u.sel(time=u.time.dt.hour==hour), v.sel(time=v.time.dt.hour==hour), suffix=f'_hour_{hour}')
            elif variable in ['moist_static_energy_3d', 'upward_air_velocity_3d']:
                print(f'plotting {variable}')
                ds = ds.compute()
                plot_spatial(ds.sel(pressure=500), opts, exps, dom, datapath, cycle_path, coarsen=False, suffix=f'_500hPa')
                plot_spatial(ds.sel(pressure=850), opts, exps, dom, datapath, cycle_path, coarsen=False, suffix=f'_850hPa')
                plot_vertical_velocity_profiles(ds, opts, exps, dom, variable, suffix='')
            else:
                if variable in ['moisture_convergence_emma']:
                    # change units to from mm/day to mm over simulation period
                    days_in_sim = ds[exp].time.size // 24
                    for exp in exps:
                        ds[exp] = -ds[exp] * days_in_sim  # convert from mm/day to total and divergence to convergence
                        ds[exp].attrs['units'] = 'mm'
                ds = ds.compute()
                plot_spatial(ds, opts, exps, dom, datapath, cycle_path, coarsen=False, suffix=f'')
                plot_spatial(ds, opts, exps, dom, datapath, cycle_path, coarsen=True, suffix= f'_coarsened')
                if 'time' in ds.dims:
                    plot_cumsum(ds, opts, exps, datapath, cycle_path, plotpath, dom)

        # Clear cache at the end of each domain to free up memory
        data_cache.clear()
        print(f"Cleared cache after processing domain {dom}")

    print(f"\nCompleted processing cylc_id: {cylc_id}")

def get_exp_path(exp, cycle, variable, cycle_path):

    exp_paths = {
        f'CCIv2_GAL9': f'{cycle_path}/{cycle}/NA/0p11/GAL9/um',
        f'CCIv2_GAL9_mod': f'{cycle_path}/{cycle}/NA/0p11/GAL9_mod/um',
        f'CCIv2_RAL3P2': f'{cycle_path}/{cycle}/NA/0p04/RAL3P2/um',
        f'CCIv2_RAL3P2_mod': f'{cycle_path}/{cycle}/NA/0p04/RAL3P2_mod/um',
    }

    if variable == 'surface_roughness_length':
        # replace um in exp_path items with ics
        for key in exp_paths:
            exp_paths[key] = exp_paths[key].replace('/um', '/ics')

    return exp_paths[exp]

def plot_cumsum(ds, opts, exps, datapath, cycle_path, plotpath, dom):
    '''plots the two experiments and a cumulative sum of variable over time'''

    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Plot cumulative sum for each experiment
    for exp in exps:
        # mask over land for this experiment
        mask_opts = get_variable_opts('land_sea_mask')
        lsm_mask = get_exp_cycles('land_sea_mask', exp, opts=mask_opts, datapath=datapath, cycle_path=cycle_path).isel(time=0).compute()
        
        ds_mask = ds[exp].where(lsm_mask == 1)
        
        # calculate spatial mean over land first, then cumulative sum
        ds_spatial_mean = ds_mask.mean(dim=['latitude', 'longitude']).compute()
        
        # calculate cumulative sum of rainfall
        pr_cumsum = ds_spatial_mean.cumsum(dim='time') * 3600  # convert from kg m-2 s-1 to mm/hour
        
        # Plot time series
        pr_cumsum.plot(ax=ax, label=exp, linewidth=2)
    
    # Format the plot
    ax.set_xlabel('')
    ax.set_ylabel(f'Cumulative [{opts["units"]}]')
    ax.set_title(f'Cumulative {opts["plot_title"]} over land - {dom}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    fname = f'{plotpath}/{opts["plot_fname"]}_cumsum_{dom}.png'
    print(f'saving cumulative plot to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    
    return

def plot_rain_diff(ds, exps, opts, plotpath, dom, coarsen=False, suffix=''):
    '''plots the two experiments and a difference between them'''

    # calculate mean mm over simulation
    pr_mean = ds.sum(dim='time') * 3600  # convert from kg m-2 s-1 to mm 

    pr_mean['diff'] = pr_mean[exps[1]] - pr_mean[exps[0]]
    pr_mean = pr_mean.compute()

    if coarsen:
        pr_mean = pr_mean.coarsen({'longitude': 10, 'latitude': 10}, boundary='trim').mean()

    ######

    proj = ccrs.PlateCarree()
    plt.close('all')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})

    pr_max = pr_mean.to_array().max().compute().values
    pr_vmax = int(np.round(pr_max / 200) * 200)
    pr_vmax = 100 if pr_vmax == 0 else pr_vmax

    for ax, exp in zip(axes.flatten(), exps+['diff']):
        cmap = opts['cmap'] if exp != 'diff' else 'coolwarm_r'
        vmin = 0 if exp != 'diff'  else -pr_vmax/5
        vmax = pr_vmax if exp != 'diff' else pr_vmax/5
        title = exp if exp != 'diff' else f'{exps[1]} - {exps[0]} difference'
        extend = 'both' if vmin < 0 else 'max'
        cbar_title =  'total precipitation [mm]' if exp != 'diff' else 'difference [mm]'
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

    fname = f'{plotpath}/{opts["plot_fname"]}_diff_{dom}{suffix}_total.png'
    print(f'saving figure to {fname}')

    fig.savefig(fname, dpi=300, bbox_inches='tight')

def plot_spatial(dss, opts, exps, dom, datapath, cycle_path, coarsen=False, suffix=''):
    '''plots the two experiments and a difference between them'''

    if 'time' in dss.dims:
        ds_mean = dss.mean(dim='time')
    else:
        ds_mean = dss.copy()

    ds_mean['diff'] = ds_mean[exps[1]] - ds_mean[exps[0]]
    ds_mean = ds_mean.compute()

    diff_max = ds_mean['diff'].max().values
    diff_min = ds_mean['diff'].min().values
    diff_vmax = max(abs(diff_min), abs(diff_max))

    if coarsen:
        ds_mean = ds_mean.coarsen({'longitude': 10, 'latitude': 10}, boundary='trim').mean()

    ######

    proj = ccrs.PlateCarree()
    plt.close('all')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})

    for ax, exp in zip(axes.flatten(), exps+['diff']):
        cmap = opts['cmap'] if exp != 'diff' else 'coolwarm'
        vmin = opts['vmin'] if exp != 'diff' else -diff_vmax
        vmax = opts['vmax'] if exp != 'diff' else diff_vmax
        title = exp if exp != 'diff' else f'{exps[1]} - {exps[0]} difference'
        cbar_title =  f'mean {opts["plot_title"]} [{opts["units"]}]' if exp != 'diff' else f'difference {opts["plot_title"]} [{opts["units"]}]'

        if opts['variable'] in ['moisture_convergence_old','moisture_convergence_emma']:
            diff_vmax = 400
            if exp == 'diff':
                cmap = 'coolwarm_r'

        levels = np.linspace(vmin, vmax, 11)

        im = ds_mean[exp].plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels,
                    extend='both',add_colorbar=False, transform=proj)
                           
        ax.set_title(title)

        # add cylc_id as text
        ax.text(0.02, 0.98, cylc_id, transform=ax.transAxes, 
                fontsize=6, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
                zorder=10)

        # # for cartopy
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds(ds_mean)
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

def plot_wind_quiver(u,v,suffix=''):

    proj = ccrs.PlateCarree()
    plt.close('all')
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                            sharex=True, sharey=True,
                            subplot_kw={'projection': proj})


    u['diff'] = u[exps[1]] - u[exps[0]]
    v['diff'] = v[exps[1]] - v[exps[0]]

    for ax, exp in zip(axes.flatten(), exps+['diff']):

        u_mean = u[exp].mean('time').persist()
        v_mean = v[exp].mean('time').persist()
        wind_speed_mean = xr.DataArray(np.sqrt(u[exp]**2 + v[exp]**2)).mean('time').persist()
        wind_speed_mean.attrs['units'] = 'm/s'

        # cmap = opts['cmap'] if exp != 'diff' else 'coolwarm'
        # vmin = opts['vmin'] if exp != 'diff' else -opts['vmax']/2
        # vmax = opts['vmax'] if exp != 'diff' else opts['vmax']/2
        title = exp if exp != 'diff' else f'{exps[1]} - {exps[0]} difference'
        cbar_title =  f'mean {opts["plot_title"]} [{opts["units"]}]' if exp != 'diff' else f'difference {opts["plot_title"]} [{opts["units"]}]'

        im = wind_speed_mean.plot(ax=ax,vmin=0,vmax=10,add_colorbar=False, transform=proj)

        ds = xr.Dataset({'u':u_mean,'v':v_mean}).rolling(longitude=20,latitude=20).mean()
        
        ds.isel(longitude=slice(0,-1,20),latitude=slice(0,-1,20)).plot.quiver(ax=ax, x='longitude',y='latitude',u='u',v='v',scale=40,transform=proj)

        ax.set_title(title)
        # # for cartopy
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(resolution='10m', color='0.1', linewidth=1, zorder=5)
        left, bottom, right, top = get_bounds(wind_speed_mean)
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

def calc_cross_correlation(var1, var2):
    """
    Calculate instantaneous correlation between two variables at each grid point
    
    Parameters:
    var1, var2: xarray DataArrays with time dimension
    
    Returns:
    correlation: DataArray with correlation coefficients
    lag: DataArray with zeros (for compatibility)
    """
    
    print(f'Calculating instantaneous correlation between variables...')
    
    # Ensure same time dimension
    var1_aligned, var2_aligned = xr.align(var1, var2, join='inner')
    
    # Use xarray's built-in correlation function which is much faster
    # This computes Pearson correlation coefficient at lag=0
    correlation = xr.corr(var1_aligned, var2_aligned, dim='time')
    
    correlation.attrs['units'] = 'correlation coefficient'
    
    return correlation

# Cache for loaded data to avoid repeated loading
data_cache = {}

def get_cached_data(variable, exp, opts, datapath, cycle_path):
    """Get data with caching to avoid repeated loads"""
    cache_key = f"{variable}_{exp}_{opts['plot_fname']}"
    
    if cache_key not in data_cache:
        print(f"Loading {cache_key} (not in cache)")
        data_cache[cache_key] = get_exp_cycles(variable, exp, opts, datapath, cycle_path)
    else:
        print(f"Using cached {cache_key}")
    
    return data_cache[cache_key]

def get_exp_cycles(variable, exp, opts, datapath, cycle_path):
    '''gets all cycles for a given experiment and variable, saves netcdf and returns xarray dataset'''

    # first check if a netcdf file has already been created
    ncfname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
    if os.path.exists(ncfname) and not overwrite_nc:
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

        exp_path = get_exp_path(exp, cycle, variable, cycle_path)

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

        da = get_um_data(variable,exp_path,opts)

        if da is None:
            print(f'WARNING: no data found at {cycle}')
        else:
            da_list.append(da)
        
        # for time invarient variables (land_sea_mask, surface_altitude) only get the first cycle
        if variable in ['land_sea_mask', 'surface_altitude','surface_roughness_length']:
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

def get_um_data(variable,exp_path,opts):
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
    if ('time' not in da.dims) and ('dim_0' in da.dims): #(variable not in ['land_sea_mask','surface_altitude','surface_roughness_length']):
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
    cbar.formatter.set_powerlimits((-6, 6))  # Use scientific notation for small/large numbers
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
    varname = opts['plot_fname']
    ds = das.to_dataset(name=varname)
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
    write_job = ds.to_netcdf(ncfname, unlimited_dims='time', encoding=encoding, compute=False)

    total_bytes = ds[varname].nbytes
    total_gb = total_bytes / 1024**3
    print(f'writing (one-shot): {ncfname}  size≈{total_gb:.2f} GB')
    write_job.compute()

    return

def calc_moisture_convergence(u, v, q):
    """
    Calculate moisture convergence from zonal and meridional wind components
    and specific humidity, accounting for spherical geometry.

    Parameters:
    u : xarray.DataArray
        Zonal wind component (m/s).
    v : xarray.DataArray
        Meridional wind component (m/s).
    q : xarray.DataArray
        Specific humidity (kg/kg).
    Returns:
    moisture_convergence : xarray.DataArray
        Moisture convergence (mm/day).
    """

    print('Calculating moisture convergence...')
    
    # Ensure input DataArrays have the same dimensions
    assert u.shape == v.shape == q.shape, "Input DataArrays must have the same dimensions."
    
    # Earth's radius in meters
    r = 6371000.0

    # Convert degrees to radians for latitude
    lat_rad = np.deg2rad(u.latitude)

    # Calculate grid spacing in meters
    dx = r * np.cos(lat_rad) * np.deg2rad(u.longitude.diff('longitude').mean())
    dy = r * np.deg2rad(u.latitude.diff('latitude').mean())

    # Calculate q*u and q*v
    qu = q * u
    qv = q * v

    # Take derivatives with respect to x (longitude) and y (latitude)
    dqu_dx = qu.differentiate('longitude') / dx
    dqv_dy = qv.differentiate('latitude') / dy

    # Moisture convergence: -div(q*v)
    moisture_convergence = -(dqu_dx + dqv_dy)

    # Convert to mm/day
    moisture_convergence = moisture_convergence * 86400  # kg/(m^2 s) to mm/day
    moisture_convergence.attrs['units'] = 'mm/day'

    return moisture_convergence

def calc_moisture_convergence_emma(u,v,qfluxu,qfluxv):

    print('Calculating moisture convergence...')

    # Ensure input DataArrays have the same dimensions
    assert u.shape == v.shape == qfluxu.shape == qfluxv.shape, "Input DataArrays must have the same dimensions."

    r = 6371000
    dx = r * np.cos(u.latitude * np.pi / 180.0) * np.pi / 180.0
    dy = r * np.pi / 180.0

    # sfcDiv = ( v.differentiate('latitude') / dy 
    #          + u.differentiate('longitude') / dx).to_dataset(name='sfcDiv')
    qDiv = ( qfluxv.differentiate('latitude') / dy 
           + qfluxu.differentiate('longitude') / dx).to_dataset(name='qDiv')

    # units
    # convert to mm/day
    qDiv = qDiv * 86400  # kg/(m^2 s) to mm/day
    qDiv['qDiv'].attrs['units'] = 'mm/day'

    return qDiv

def calc_moist_static_energy(t3d, rh3d, z3d):
    """
    Calculate moist static energy from 3D temperature, relative humidity, and geopotential height.
    
    Moist static energy (MSE) is defined as:
    MSE = cp*T + g*z + Lv*q
    
    Where specific humidity q is calculated from relative humidity using:
    q = (RH/100) * qs
    qs = 0.622 * es / (p - 0.378 * es)  (saturation mixing ratio)
    es = 6.112 * exp(17.67 * (T-273.15) / (T-29.65))  (saturation vapor pressure, Magnus formula)
    
    Parameters:
    -----------
    t3d : xarray.DataArray
        3D air temperature (K) with pressure coordinate
    rh3d : xarray.DataArray  
        3D relative humidity (%) with pressure coordinate
    z3d : xarray.DataArray
        3D geopotential height (m) with pressure coordinate
        
    Returns:
    --------
    mse : xarray.DataArray
        Moist static energy (J/kg)
    """
    
    print('Calculating moist static energy...')
    
    # Ensure all inputs have same dimensions and pressure levels
    assert t3d.shape == rh3d.shape == z3d.shape, "Input DataArrays must have the same dimensions."
    
    # Physical constants
    cp = 1004.0    # Specific heat capacity of air at constant pressure (J/(kg·K))
    g = 9.81       # Gravitational acceleration (m/s²)
    Lv = 2.5e6     # Latent heat of vaporization (J/kg)
    
    # Ensure temperature is in Kelvin
    if t3d.attrs.get('units', '') == '°C':
        print('converting temperature from Celsius to Kelvin...')
        T_K = t3d + 273.15
    else:
        T_K = t3d.copy()
    
    # Convert relative humidity from % to fraction if needed
    if rh3d.max() > 2.0:  # Assume it's in percentage if max > 200%
        print('converting relative humidity from % to fraction...')
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
    
    # Calculate moist static energy: MSE = cp*T + g*z + Lv*q
    mse = cp * T_K + g * z3d + Lv * q
    
    # Set attributes
    mse.attrs['units'] = 'J/kg'
    mse.attrs['long_name'] = 'moist static energy'
    mse.attrs['description'] = 'MSE = cp*T + g*z + Lv*q'
    
    print(f'MSE range: {mse.min().values:.0f} to {mse.max().values:.0f} J/kg')
    
    return mse

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
        'dtype'     : 'float32',
        'variable'  : variable,
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

    if variable == 'upward_air_velocity_3d':
        opts.update({
            'constraint': 'm01s15i242',
            'plot_title': 'upward air velocity 3d',
            'plot_fname': 'upward_air_velocity_3d',
            'units'     : 'm s-1',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverc',
            'vmin'      : -1,
            'vmax'      : 1,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    if variable == 'upward_air_velocity_3d_over_land_spatial_mean':
        opts.update({
            'constraint': 'm01s15i242',
            'plot_title': 'upward air velocity 3d over land spatial mean',
            'plot_fname': 'upward_air_velocity_3d_over_land_spatial_mean',
            'units'     : 'm s-1',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverc',
            'vmin'      : -1,
            'vmax'      : 1,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    if variable == 'wet_bulb_potential_temperature_3d':
        opts.update({
            'constraint': 'm01s16i205',
            'plot_title': 'wet bulb potential temperature 3d',
            'plot_fname': 'wet_bulb_potential_temperature_3d',
            'units'     : 'K',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverc',
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    if variable == 'geopotential_height_3d':
        opts.update({
            'constraint': 'm01s16i202',
            'plot_title': 'geopotential height 3d',
            'plot_fname': 'geopotential_height_3d',
            'units'     : 'm',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverd',
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    if variable == 'air_temperature_3d':
        opts.update({
            'constraint': 'm01s16i203',
            'plot_title': 'air temperature 3d',
            'plot_fname': 'air_temperature_3d',
            'units'     : 'K',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverd',
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    if variable == 'relative_humidity_3d':
        opts.update({
            'constraint': 'm01s16i204',
            'plot_title': 'relative humidity 3d',
            'plot_fname': 'relative_humidity_3d',
            'units'     : '%',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverd',
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

    elif variable == 'wind_speed_of_gust':
        opts.update({
            'constraint': 'm01s03i463',
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

    elif variable == 'wind_quiver':
        opts.update({
            'units'     : 'm/s',
            'vmin'      : 0,
            'vmax'      : 10,
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

    elif variable == 'low_cloud_amount_instant':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s09i203', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) not in cube.cell_methods),
            'plot_title': 'low cloud amount (instant)',
            'plot_fname': 'low_cloud_amount_instant',
            'units'     : '-',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'low_cloud_amount_mean':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s09i203', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'low cloud amount (mean)',
            'plot_fname': 'low_cloud_amount_mean',
            'units'     : '-',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
    
    elif variable == 'medium_cloud_amount_instant':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s09i204', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) not in cube.cell_methods),
            'plot_title': 'medium cloud amount (instant)',
            'plot_fname': 'medium_cloud_amount_instant',
            'units'     : '-',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
    
    elif variable == 'medium_cloud_amount_mean':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s09i204', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'medium cloud amount (mean)',
            'plot_fname': 'medium_cloud_amount_mean',
            'units'     : '-',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'high_cloud_amount_instant':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s09i205', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) not in cube.cell_methods),
            'plot_title': 'high cloud amount (instant)',
            'plot_fname': 'high_cloud_amount_instant',
            'units'     : '-',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
    
    elif variable == 'high_cloud_amount_mean':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s09i205', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'high cloud amount (mean)',
            'plot_fname': 'high_cloud_amount_mean',
            'units'     : '-',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
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

    elif variable == 'moisture_convergence_emma':
        opts.update({
            'vmin'      : -20,
            'vmax'      : 20,
            'units'     : 'mm/day',
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'moisture_convergence_old':
        opts.update({
            'vmin'      : -20,
            'vmax'      : 20,
            'units'     : 'mm/day',
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'land_sea_mask':
        opts.update({
            'constraint': 'land_binary_mask',
            'stash'     : 'm01s00i030',
            'plot_title': 'land sea mask',
            'plot_fname': 'land_sea_mask',
            'units'     : 'm',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 1,
            'fmt'       : '{:.1f}',
            'dtype'     : 'int16',
            })

    elif variable == 'surface_roughness_length':
        opts.update({
            'constraint': 'surface_roughness_length',
            'stash'     : 'm01s00i026',
            'plot_title': 'surface roughness length',
            'plot_fname': 'surface_roughness_length',
            'units'     : 'm',
            'fname'     : 'umnsaa_da',
            'vmin'      : 0,
            'vmax'      : 1,
            'fmt'       : '{:.2f}',
            'dtype'     : 'int16',
            })

    elif variable == 'boundary_layer_depth':
        opts.update({
            'constraint': 'm01s00i025',
            'plot_title': 'boundary layer depth',
            'plot_fname': 'boundary_layer_depth',
            'units'     : 'm',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 3000,
            'fmt'       : '{:.1f}',
            'dtype'     : 'int16',
            })

    elif variable == 'precip_w_cross_correlation':
        opts.update({
            'plot_title': 'precipitation - vertical wind cross-correlation',
            'plot_fname': 'precip_w_cross_correlation',
            'units': 'correlation coefficient',
            'vmin': -1,
            'vmax': 1,
            'cmap': 'coolwarm',
            'fmt': '{:.3f}',
        })

    elif variable == 'moist_static_energy_3d':
        opts.update({
            'plot_title': 'moist static energy 3d',
            'plot_fname': 'moist_static_energy_3d',
            'units': 'J/kg',
            'vmin': 300000,
            'vmax': 380000,
            'cmap': 'plasma',
            'fmt': '{:.0f}',
            'dtype': 'float32',
        })

    # add variable to opts
    opts.update({'variable':variable})

    return opts

def plot_vertical_velocity_profiles(ds,opts,exps,dom,variable,suffix=''):
    # mask to land areas only
    ds_mean = xr.Dataset()
    for exp in exps:
        print(f'processing {exp} {variable}')

        mask_opts = get_variable_opts('land_sea_mask')
        lsm_mask = get_exp_cycles('land_sea_mask', exp, opts=mask_opts, datapath=datapath, cycle_path=cycle_path).isel(time=0).compute()
        da_mean = ds[exp].where(lsm_mask == 1)
        ds_mean[exp] = da_mean.mean(dim=['latitude','longitude']).compute()
    
    if 'time' in ds_mean.dims:
        ds_time_mean = ds_mean.mean(dim='time').compute()
    else:
        ds_time_mean = ds_mean.compute()

    # create scatter plot with pressure on y axis and upward air velocity on x axis
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 6))
    
    # Plot each experiment
    for exp in exps:
        pres = ds_time_mean[exp].pressure.values
        ax.plot(ds_time_mean[exp].values, pres, 
               label=exp, linewidth=2)
        ax.set_yticks(pres)
        ax.set_xlabel(f'{opts["plot_title"]} [{opts["units"]}]')
        ax.set_ylabel('Pressure [hPa]')

    # Invert y-axis so 1000 hPa is at bottom and lower pressures at top
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{opts["plot_title"]} Profile over land - {dom}')
    
    # Save the plot
    fname = f'{plotpath}/{opts["plot_fname"]}_profile_{dom}{suffix}.png'
    print(f'saving profile plot to {fname}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create plots from UM files')
    parser.add_argument('--cylc-id', '--cylc_id', type=str, nargs='+', 
                        default=['rns_ostia_NA_2019'],
                        help='Cylc ID(s) to process (default: rns_ostia_NA_2019)')
    return parser.parse_args()

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

    # Parse command line arguments
    args = parse_arguments()
    cylc_ids = args.cylc_id if isinstance(args.cylc_id, list) else [args.cylc_id]
    
    print(f"Processing cylc_id(s): {cylc_ids}")
    
    # Process each cylc_id
    for cylc_id in cylc_ids:
        print(f"\n{'='*60}")
        print(f"Processing cylc_id: {cylc_id}")
        print(f"{'='*60}")
        
        # Set paths for this cylc_id
        cycle_path = f'/scratch/fy29/mjl561/cylc-run/{cylc_id}/share/cycle'
        datapath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/netcdf'
        plotpath = f'/g/data/fy29/mjl561/cylc-run/{cylc_id}/figures'
        
        # make plotpath if it doesn't exist
        os.makedirs(plotpath, exist_ok=True)
    
        main(datapath, plotpath, cycle_path, doms, cylc_id)

        print(f"\n{'='*60}")
        print(f"Completed processing all cylc_ids: {cylc_ids}")
        print(f"{'='*60}")

