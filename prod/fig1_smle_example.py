#! /usr/bin/env python3
import os 
import sys
import numpy as np
import xarray as xr

sys.path.append('/glade/u/home/dtokuda/localpy/CMasher')
import cmasher as cmr

from ncar import CMIP as cmip
from laiapy import csm
import figlib
import pdutil
import matplotlib.pyplot as plt

import lsmconst as ll

sim_cond = {
    'experiment_id': 'historical',
    'source_id': 'IPSL-CM6A-LR',
    'member_id': 'r1i1p1f1',
    'grid_label': '2x2',
}
start_year, end_year = 1903, 2012

sm_scale = 1e-2
le_scale = None


lat, lon = 11, 20

# ===================================================================================================
def count_model_params(model):
    for i in range(10):
        key = (model, 'param', i)
        if not key in csm.netcdf.DICT_NETCDF_INFO: return i

def get_params(ds, model, lat, lon):
    n_param = count_model_params(model)
    params = []
    for i in range(n_param):
        var_name = csm.netcdf.DICT_NETCDF_INFO[(model, 'param', i)][0]
        da = ds[var_name]
        param = da.sel(lat=lat, lon=lon, method='nearest').values
        params += [param]
    return params

# ---------------------------------------------------------------------------------------------------
def scatter(ax, x, y, xmin, xmax, dx, ymin, ymax, dy):
    xbins = np.arange(xmin, xmax + dx, dx)
    ybins = np.arange(ymin, ymax + dy, dy)
    h, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
    h = np.ma.masked_less_equal(h, 0)
    h /= len(x)
    #h /= (len(x) * 0.01) # (%)
    X, Y = np.meshgrid(xbins, ybins)

    cmap = cmr.get_sub_cmap('Reds', 0.1, 0.8)
    im = ax.pcolormesh(X, Y, h.T, cmap=cmap,
        vmin=0, vmax=0.01)
    return im

def add_indicators(ax, xmin, xmax, bp0x, bp0y, bp1x, bp1y, **kwargs):
    #dy = 50
    #ax.vlines(bp0x, bp0y, bp0y + dy, **kwargs)
    #ax.text(bp0x + 0.03, bp0y + dy, 'Wilting point',
    #    ha='right', va='bottom')

    #dy = 35
    #ax.vlines(bp1x, bp1y - dy, bp1y, **kwargs)
    #ax.text(bp1x, bp1y - dy, 'Critical SM',
    #    ha='left', va='top')

    x0 = 0.13
    x0, x1 = min(x0, bp1x), max(x0, bp1x)
    ax.hlines(bp1y, x0, x1, **kwargs)
    ax.text(x0, bp1y, 'Saturated LE',
        ha='right', va='center')
    ax.text(0.12, 0.5 * (bp0y + bp1y), 'SM-LE slope',
        ha='right', va='center')

def add_regimes(ax0, ax1, xmin, xmax, bp0x, bp0y, bp1x, bp1y, **kwargs):
    _bp0x = (bp0x - xmin) / (xmax - xmin)
    _bp1x = (bp1x - xmin) / (xmax - xmin)
    ax0.vlines(bp0x, 0, bp0y, **kwargs)
    ax0.vlines(bp1x, 0, bp1y, **kwargs)
    ax1.vlines(_bp0x, 0.6, 1, **kwargs)
    ax1.vlines(_bp1x, 0.6, 1, **kwargs)

    y, bufy = 0.3, 0.1

    ax1.plot([0, 1], [y, y],
        ls='-', color='k', lw=1, zorder=99)
    ax1.plot([_bp0x, _bp1x], [y, y],
        ls='-', marker='|', color='k', lw=1, zorder=99)
    ax1.text(_bp0x, y + bufy, 'Wilting point',
        ha='center', va='bottom')
    ax1.text(_bp1x, y + bufy, 'Critical SM',
        ha='center', va='bottom')

    x = 0.5 * (0 + _bp0x)
    ax1.text(x, y - bufy, 'Dry regime',
        ha='center', va='top')
    x = 0.5 * (_bp0x + _bp1x)
    ax1.text(x, y - bufy, 'Transitional regime',
        ha='center', va='top')
    x = 0.5 * (_bp1x + 1)
    ax1.text(x, y - bufy, 'Wet regime',
        ha='center', va='top')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

# ===================================================================================================
def main(*args):
    time_range = cmip.dtutil.ints2tr(start_year, 1, end_year, 12)

    '''Input data'''
    cmip_catalog = cmip.catalog.read('local')
    cmip_catalog = pdutil.filter(cmip_catalog, **sim_cond)
    (sm_table, sm_variable), (le_table, le_variable) = ll.config.get_table_variable_ids('sm', 'le')

    sm_ds = cmip.read_nc(cmip_catalog, time_range,
        table_id=sm_table, variable_id=sm_variable)
    le_ds = cmip.read_nc(cmip_catalog, time_range,
        table_id=le_table, variable_id=le_variable)

    '''Estimated regime parameters'''
    param_catalog = ll.catalog.read('csm_monthly')
    param_catalog = pdutil.filter(param_catalog,
        **sim_cond, time_range=str(time_range))
    assert(len(param_catalog) == 1)
    path = param_catalog.iloc[0]['path']
    param_ds = xr.open_dataset(path)

    prop_catalog = ll.catalog.read('prop_monthly')
    prop_catalog = pdutil.filter(prop_catalog,
        **sim_cond, time_range=str(time_range))
    assert(len(prop_catalog) == 1)
    path = prop_catalog.iloc[0]['path']
    prop_ds = xr.open_dataset(path)

    best_model = prop_ds['model']
    print(csm.model.MODELS[int(best_model.sel(lat=lat, lon=lon, method='nearest').values)])
    sm_arr = sm_ds[sm_variable].sel(lat=lat, lon=lon, method='nearest').to_masked_array()
    if sm_scale is not None: sm_arr *= sm_scale
    le_arr = le_ds[le_variable].sel(lat=lat, lon=lon, method='nearest').to_masked_array()
    if le_scale is not None: le_arr *= le_scale









    #xmin, xmax, dx = 0.20, 0.50, 0.01
    xmin, xmax, dx = 0.0, 0.30, 0.005
    #xmin, xmax, dx = 0.0, 0.50, 0.01
    #ymin, ymax, dy = 20, 140, 4
    ymin, ymax, dy = 0, 130, 2.5
    #ymin, ymax, dy = 0, 140, 4

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes((0, 0.2, 1, 0.8))
    ax0 = figlib.axesutil.add_axes(ax, (0, 0, 0.8, 1))
    cax = figlib.axesutil.add_axes(ax, (0.85, 0, 0.05, 1))
    ax.remove()
    ax1 = fig.add_axes((0, 0, 0.8, 0.2))
    ax1.axis('off')
    im = scatter(ax0, sm_arr, le_arr, xmin, xmax, dx, ymin, ymax, dy)

    model = '111'
    params = get_params(param_ds, model, lat, lon)
    csm.figure.plot(ax0, model, params)

    bp0x, bp0y, _, bp1x, bp1y = params
    #csm.figure.scatter_sm_le(ax, sm, le)
    #csm.figure.plot_001(ax, param001)
    #csm.figure.plot_010(ax, param010)
    #csm.figure.plot_110(ax, param110)
    #csm.figure.plot_011(ax, param011)
    #csm.figure.plot_111(ax, params)
    kwargs = {'color': 'grey', 'lw': 1, 'ls': '--'}
    add_indicators(ax0, xmin, xmax, bp0x, bp0y, bp1x, bp1y, **kwargs)

    #plt.draw()
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    ax0.text(1.05, -0.04, f'({ll.figure.SM_UNIT})',
        ha='left', va='center', transform=ax0.transAxes)

    ax0.text(0, 1, f'({ll.figure.LE_UNIT})',
        ha='right', va='top', transform=ax0.transAxes)

    add_regimes(ax0, ax1, xmin, xmax, bp0x, bp0y, bp1x, bp1y, **kwargs)
    #ax.set_xlim(0, 0.5)
    plt.colorbar(im, cax=cax, extend='max')

    ll.prod.savefig()
    plt.show()
    plt.close()
    return

if __name__=='__main__':
    main(*sys.argv)