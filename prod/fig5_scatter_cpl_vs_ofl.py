#! /usr/bin/env python3
import os 
import sys
import numpy as np
from scipy.stats import gaussian_kde

sys.path.append('/glade/u/home/dtokuda/localpy/CMasher')
import cmasher as cmr

from ncar import CMIP as cmip
from laiapy import csm
import figlib
import matplotlib.pyplot as plt
from util import stat as statutil

import lsmconst as ll

# ===================================================================================================
def save(*args):
    import xarray as xr
    from util import stat as statutil

    dir_name = 'prop_enstat_monthly'
    cpl_cond = {
        'member_id': '*members',
        'grid_label': '2x2',
        'time_range': '190301-201212',
    }
    #cpl_ens_attr_name = 'member_id'
    #cpl_min_members = 10

    ofl_dir_name = 'prop_monthly'
    ofl_cond = {
        'source_id': '~GISS-E2-1-G',
        #'experiment_id': '*members',
        'experiment_id': 'land-hist',
        'grid_label': '2x2',
        'time_range': '190301-201212',
    }

    loop_attr_name = 'source_id'

    catalog = ll.catalog.read(dir_name)
    cpl_catalog = cmip.catalog.filter(catalog, **cpl_cond)

    catalog = ll.catalog.read(ofl_dir_name)
    ofl_catalog = cmip.catalog.filter(catalog, **ofl_cond)
    ofl_catalog = ll.catalog.filter_single_member(ofl_catalog)

    data = {}
    for loop_attr_value in ll.CPLOFL_SOURCE_IDS:
        _cpl_catalog = cmip.catalog.filter(cpl_catalog, log=False, **{loop_attr_name: loop_attr_value})
        assert(len(_cpl_catalog) == 1)
        _ofl_catalog = cmip.catalog.filter(ofl_catalog, log=False, **{loop_attr_name: loop_attr_value})
        assert(len(_ofl_catalog) == 1)

        cpl_path = _cpl_catalog.iloc[0]['path']
        cpl_member_num = ll.catalog.get_member_num(_cpl_catalog.iloc[0]['member_id'])
        ofl_path = _ofl_catalog.iloc[0]['path']
        print(cpl_path)
        print(ofl_path)

        cpl_ds = xr.open_dataset(cpl_path)
        ofl_ds = xr.open_dataset(ofl_path)
        for prop_name in ll.PROP_NAMES:
            cpl_arr = cpl_ds[prop_name + '_mean'].to_masked_array().flatten()
            cpl_num = cpl_ds[prop_name + '_num'].to_masked_array().flatten()
            cpl_arr = np.ma.masked_array(cpl_arr, mask=cpl_num <= 0.5 * cpl_member_num)
            ofl_arr = ofl_ds[prop_name].to_masked_array().flatten()

            data[(prop_name, loop_attr_value, 'JC')] = statutil.jaccard(cpl_arr, ofl_arr)
            cpl_arr, ofl_arr = statutil.filter_invalid(cpl_arr, ofl_arr)
            data[(prop_name, loop_attr_value, 'n')] = len(cpl_arr)
            data[(prop_name, loop_attr_value, 'CC')] = statutil.corr(cpl_arr, ofl_arr)
            data[(prop_name, loop_attr_value, 'RMSD')] = statutil.rmsd(cpl_arr, ofl_arr)

            data[(prop_name, loop_attr_value, 'arrs')] = (cpl_arr, ofl_arr)

    ll.prod.dump_pickle(data)
    return

DICT_PROP_HISTMAX = {
    'wilt': 0.05,
    'csm': 0.015,
    'lesat': 0.01,
    'slope': 3e-3,
}

# ===================================================================================================
def scatter_same_lim(ax, x, y, prop_name,
    texts=None, *args, **kwargs):
    x, y = statutil.filter_invalid(x, y)
    #ax.scatter(x, y, *args, **kwargs, zorder=9)
    #h, _, _, _ = ax.hist2d(x, y, bins=[50, 50],
    #    cmin=1, density=True)

    if False:
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        im = ax.scatter(x, y, c=z, s=50, cmap='hot')
    vmin, vmax = ll.prod.DICT_MEAN_RANGE[prop_name]
    bins = np.linspace(vmin, vmax, 51)
    h, _, _ = np.histogram2d(x, y, bins=(bins, bins))
    h = np.ma.masked_less_equal(h, 0)
    h /= len(x)
    X, Y = np.meshgrid(bins, bins)
    #h = gaussian_kde(h.flatten())(h.flatten()).reshape(50, 50)
    #idx = z.argsort()
    #x, y, z = x[idx], y[idx], z[idx]
    #im = ax.scatter(x, y, c=z, s=50, cmap='hot')

    cmin, cmax = 0, DICT_PROP_HISTMAX[prop_name]
    cmap = cmr.get_sub_cmap('Reds', 0, 0.8)
    im = ax.pcolormesh(X, Y, h.T, cmap=cmap,
        vmin=cmin, vmax=cmax)

    figlib.util.text_loc(ax, texts, loc='lower right')

    figlib.util.scatter.plot_45line(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.hline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.vline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.set_xylim(ax, vmin, vmax)
    return im

# ---------------------------------------------------------------------------------------------------
def draw(*args):
    import matplotlib.pyplot as plt

    import figlib

    source_ids = ll.CPLOFL_SOURCE_IDS

    #width_ax, height_ax = 3, 3
    width_ax, height_ax = 2, 2
    nrows, ncols = 4, len(source_ids)

    #figsize = (width_ax * ncols, height_ax * nrows)
    gsfig = figlib.GridSpecFig(nrows=nrows, ncols=ncols, axsize=(width_ax, height_ax), wspace=0.15, hspace=0.3)

    data = ll.prod.load_pickle()
    for irow, prop_name in enumerate(ll.PROP_NAMES):
        for icol, source_id in enumerate(source_ids):
            ax = gsfig[irow, icol]
            x, y = data[(prop_name, source_id, 'arrs')]

            text = []
            for stat_name in ('n', 'CC', 'RMSD'):
                if stat_name == 'n':
                    text += [f'{stat_name} = {data[(prop_name, source_id, stat_name)]}']
                else:
                    text += [f'{stat_name} = {data[(prop_name, source_id, stat_name)]:.2f}']
            im = scatter_same_lim(ax, x, y, prop_name,
                texts=text, color='royalblue', alpha=0.2)

            if gsfig.is_right(icol):
                _ax = figlib.axesutil.add_axes(ax, (1.05, 0, 0.05, 1))
                cbar = plt.colorbar(im, cax=_ax)
                cbar.ax.ticklabel_format(style='sci', scilimits=(-3,3)) 


    for irow in range(nrows):
        for icol in range(1, ncols):
            ax = gsfig[irow, icol]
            ax.set_yticks([])

    # source_id
    irow = 0
    for icol, source_id in enumerate(source_ids):
        text = f' ({("i", "ii", "iii", "iv", "v", "vi", "vii")[icol]}) {source_id}'
        #if icol == 0:
        #    text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAMEUNIT[ll.PROP_NAMES[irow]]}\n' + text
        ax = gsfig[irow, icol]
        #ax.text(-0.02, 1.05, text, va='bottom', ha='left',
        #    rotation='horizontal', rotation_mode='anchor',
        #    transform=ax.transAxes)
        ax.set_title(text)

    # prop_names
    icol = 0
    for irow, prop_name in enumerate(ll.PROP_NAMES):
        text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAMEUNIT[prop_name]}'
        ax = gsfig[irow, icol]
        #ax.text(-0.02, 1.05, text, va='bottom', ha='left',
        #    rotation='horizontal', rotation_mode='anchor',
        #    transform=ax.transAxes)
        ax.set_ylabel(text)

    ll.prod.savefig()
    plt.show()
    plt.close()
    return

if __name__=='__main__':
    #save(*sys.argv)
    draw(*sys.argv)