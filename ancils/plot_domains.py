__version__ = "2025-04-16"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

"""
Plots the ancillary domains found in the ancil_path directory.

This uses the analysis3 environment:
    module purge; module use /g/data/xp65/public/modules; module load conda/analysis3'
"""

import os
import sys
import iris
import xarray as xr
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import importlib

############## set up ##############

oshome=os.getenv('HOME')
ancil_path = f'{oshome}/cylc-run/ancils_NA_trials/share/data/ancils/NA'
plot_path = os.path.dirname(ancil_path) # parent dir of ancil_path
domains = os.listdir(ancil_path) # list of dirs in ancil_path
region = ancil_path.split("/")[-1]

############## functions ##############

def plot_domain_orography():
    """
    Plot the orography of the different domains
    Mask out the ocean with the land sea mask ancil
    """

    data = {}
    for domain in domains:
        print(f'loading {domain} data')

        # get land sea mask ancil
        opts = get_variable_opts('land_sea_mask')
        fname = f'{ancil_path}/{domain}/{opts["fname"]}'
        cb = iris.load_cube(fname, constraint=opts["constraint"])
        lsm = xr.DataArray().from_iris(cb)

        # get orography ancil
        opts = get_variable_opts('surface_altitude')
        fname = f'{ancil_path}/{domain}/{opts["fname"]}'
        cb = iris.load_cube(fname, constraint=opts["constraint"])
        # convert to xarray and constrain to lsm
        data[domain] = xr.DataArray().from_iris(cb)
        # reindex lsm (rounding errors)
        lsm = lsm.reindex_like(data[domain],method='nearest')
        data[domain] = data[domain].where(lsm>0)

    # get resolutions of domains
    rslns = {dom: data[dom].rio.resolution() for dom in domains}
    # sort domains based on their size
    doms = sorted(domains, key=lambda x: data[x].shape[0]*rslns[x][0]*data[x].shape[1]*rslns[x][1], reverse=True)

    #############################################

    print(f"plotting")

    proj = ccrs.AlbersEqualArea()
    opts = get_variable_opts('surface_altitude')
    cmap = plt.get_cmap(opts['cmap'])

    plt.close('all')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(11,9),
                            sharey=True,sharex=True,
                            subplot_kw={'projection': proj},
                            )
    # update vmax based on outer domain, to nearest 100m
    vmax = np.floor(data[doms[0]].max().values/100)*100 - 100

    for domain in doms:
        print(f'plotting {domain}')
        im = data[domain].plot(ax=ax,cmap=cmap, vmin=opts['vmin'],vmax=vmax,
            add_colorbar=False, transform=proj)
        # draw rectangle around domain
        left, bottom, right, top = get_bounds(data[domain])
        ax.plot([left, right, right, left, left], [bottom, bottom, top, top, bottom], 
            color='red', linewidth=1, linestyle='dashed' if domain=='SY_1' else 'solid')
        # label domain with white border around black text
        domain_text = f'{domain}: {data[domain].shape[0]}x{data[domain].shape[1]}'
        ax.text(right-0.1, top-0.1, f'{domain_text}', fontsize=8, ha='right', va='top', color='k',
                path_effects=[path_effects.withStroke(linewidth=1.5, foreground='w')])

    cbar_title = f"{opts['plot_title'].capitalize()} [{opts['units']}]"
    cbar = custom_cbar(ax,im,cbar_loc='right')  
    cbar.ax.set_ylabel(cbar_title)
    cbar.ax.tick_params(labelsize=8)

    # # for cartopy
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.coastlines(color='k',linewidth=0.5,zorder=5)
    left, bottom, right, top = get_bounds(data[doms[0]])
    ax.set_extent([left, right, bottom, top], crs=proj)
    
    ax = distance_bar(ax,distance=200)
    ax.set_title(f'{region} domains')

    fname = f'{plot_path}/{region}_domains_{opts["plot_fname"]}.png'
    print(f'saving {fname}')
    fig.savefig(fname,dpi=300,bbox_inches='tight')

    return

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
    Create a custom colorbar to replace the ugly cartopy one.
    """

    if cbar_loc == 'right':
        cax = inset_axes(ax,
            width='4%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, ticks = ticks)
    
    else:
        cbar_loc == 'bottom'
        cax = inset_axes(ax,
            width='100%',  # % of parent_bbox width
            height='4%',
            loc='lower left',
            bbox_to_anchor=(0, -0.15, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, orientation='horizontal', ticks = ticks)

    return cbar

def distance_bar(ax,distance=100):
    """
    Add a distance bar to the plot with geodesic distance in km
    """

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    xdist = abs(xlims[1]-xlims[0])
    offset = 0.03*xdist

    # plot distance bar
    start = (xlims[0]+offset,ylims[0]+offset)
    end = cgeo.Geodesic().direct(points=start,azimuths=90,distances=distance*1000).flatten()
    ax.plot([start[0],end[0]],[start[1],end[1]], color='0.65', linewidth=1.5)
    ax.text(start[0]+offset/7,start[1]+offset/5, f'{distance} km', 
        fontsize=9, ha='left',va='bottom', color='0.65')

    return ax

def get_variable_opts(variable):
    '''standard variable options for plotting. to be updated within master script as needed'''

    # standard opts
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
        }

    if variable == 'surface_altitude':
        opts.update({
        'constraint': 'surface_altitude',
        'units'     : 'm',
        'obs_key'   : 'None',
        'fname'     : 'qrparm.orog',
        'vmin'      : 0,
        'vmax'      : 1500,
        'cmap'      : 'terrain',
        })

    elif variable == 'land_sea_mask':
        opts.update({
            'constraint': 'm01s00i030',
            'plot_title': 'land sea mask',
            'plot_fname': 'land_sea_mask',
            'units'     : '1',
            'fname'     : 'qrparm.mask',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'viridis',
            'fmt'       : '{:.2f}',
            })

    # add variable to opts
    opts.update({'variable':variable})
    
    return opts

if __name__ == '__main__':
    plot_domain_orography()
