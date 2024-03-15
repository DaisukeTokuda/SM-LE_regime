#! /usr/bin/env python3
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/glade/u/home/dtokuda/localpy/CMasher')
import cmasher as cmr

from ncar import CMIP as cmip
import figlib

import lsmconst as ll

min_frac = 0.5

source_ids = ll.CPL_SOURCE_IDS
LEFT_KWARGS_DEF = {
    'va': 'bottom',
    'ha': 'center',
    'rotation': 'vertical',
    'rotation_mode': 'anchor',
}

# ===================================================================================================
def save(*args):
    import xarray as xr

    dir_name = 'prop_enstat_monthly'
    cond = {
        'source_id': source_ids,
        'member_id': '*members',
        'grid_label': '2x2',
        'time_range': '190301-201212',
    }
    var_names = ['wilt', 'slope', 'csm', 'lesat']
    ensemble_attr_name = 'source_id'

    catalog = ll.catalog.read(dir_name)
    catalog = cmip.catalog.filter(catalog, **cond)
    #ens_attr_name = 'member_id'
    #min_members = 10
    #catalog = ll.catalog.filter_member_num(catalog, ens_attr_name, min_members)

    dict_src = {var_name: [] for var_name in var_names}
    for attr_value in source_ids:
        _catalog = cmip.catalog.filter(catalog, log=False, **{ensemble_attr_name: attr_value})
        #print(_catalog['path'].values)
        assert(len(_catalog) == 1)
        path = _catalog.iloc[0]['path']
        print(path)

        ds = xr.open_dataset(path)
        for var_name in var_names:
            arr = ds[var_name + '_mean'].to_masked_array()
            dict_src[var_name] += [arr]

    data = {}
    for var_name in var_names:
        arr = np.ma.stack(dict_src[var_name], axis=0)
        sample_size = (~arr.mask).sum(axis=0)
        mask = (sample_size / len(arr)) < min_frac
        data[(var_name, 'mean')] = np.ma.masked_array(arr.mean(axis=0), mask=mask)
        data[(var_name, 'std')] = np.ma.masked_array(np.ma.std(arr, axis=0), mask=mask)
    ll.prod.dump_pickle(data)
    return

# ===================================================================================================
def print_vminmax(data, gsfig):
    for icol, var_name in enumerate(data['col']):
        if var_name.endswith('num'):
            vmin, vmax = 0, 1
        else:
            vmin, vmax = ll.misc.get_percentile_vminmax(gsfig.data[:, icol], ll.figure.PMIN, ll.figure.PMAX)
            print(var_name, vmin, vmax)


def colorbar(ax, im, loc='bottom', **kwargs):
    if loc == 'bottom':
        _kwargs = {'orientation': 'horizontal', 'extend': 'both'}
        pos = (0, 0.05, 1, 0.05)
    if loc == 'right':
        _kwargs = {'orientation': 'vertical', 'extend': 'both'}
        pos = (1.02, 0.1, 0.03, 0.8)
    ax_cb = figlib.axesutil.add_axes(ax, pos)
    _kwargs.update(kwargs)
    cb = plt.colorbar(im,
        cax=ax_cb, **_kwargs)
    cb.ax.tick_params(labelsize=8)
    ax_cb.set_zorder(99)
    return cb

# ---------------------------------------------------------------------------------------------------
def draw(*args):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import itertools

    import figlib

    width_ax, height_ax = 4, 2
    stat_names = ['mean', 'std']
    nrows, ncols = len(ll.PROP_NAMES), len(stat_names)

    dict_var_cmap = {
        'mean': cmr.get_sub_cmap('summer_r', 0, 0.8),
        'std': cmr.get_sub_cmap('Reds', 0, 0.8),
        'num': 'Greens',
    }

    data = ll.prod.load_pickle()
    gsfig = figlib.GridSpecFig(nrows=nrows, ncols=ncols, axsize=(width_ax, height_ax), wspace=0.08, hspace=0.1)

    titles = ['(i) Inter-model mean', '(ii) Inter-model STD']
    labels = [f'({["a", "b", "c", "d"][i]}) {ll.figure.DICT_PROP_NAME[prop_name]}\n({ll.figure.DICT_PROP_UNIT[prop_name]})'
        for i, prop_name in enumerate(ll.PROP_NAMES)]
    for (irow, prop_name), (icol, stat_name) in itertools.product(enumerate(ll.PROP_NAMES), enumerate(stat_names)):
        arr = data[(prop_name, stat_name)]
        if stat_name == 'mean':
            vmin, vmax = ll.prod.DICT_MEAN_RANGE[prop_name]
        else:
            vmin, vmax = ll.prod.DICT_STD_RANGE[prop_name]
        cmap = dict_var_cmap[stat_name]

        ax = gsfig[irow, icol]
        gsfig.option[irow, icol].update({'vmin': vmin, 'vmax': vmax, 'cmap': cmap})
        _ax, im = ll.figure.cp_imshow(ax, arr, **gsfig.option[irow, icol].data)

        '''
        text = f'({"abcd"[irow]} - {["i", "ii"][icol]})'
        _ax.text(0.035, 0.12, text,
            va='bottom', ha='left', transform=_ax.transAxes, bbox={'facecolor': 'white', 'edgecolor': 'none'},
            zorder=99)
        '''
        if gsfig.is_top(irow):
            ax.set_title(titles[icol])
        if gsfig.is_left(icol):
            #ax.set_ylabel(labels[irow])
            ax.text(0, 0.5, labels[irow],
                **LEFT_KWARGS_DEF, zorder=99)


        extend = 'both'
        if vmin == 0:
            extend = 'max'
            if vmax == 1:
                extend = 'neither'
        colorbar(ax, im, loc='bottom', extend=extend)

        #ax.remove()
        #gsfig[irow, icol] = _ax
    if False:
        icol = 0
        for irow, prop_name in enumerate(ll.PROP_NAMES):
            ax = gsfig[irow, icol]
            text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAMEUNIT[prop_name]}'
            ax.set_ylabel(text)
        irow = 0
        for icol, stat_name in enumerate(stat_names):
            ax = gsfig[irow, icol]
            text = f'({["i", "ii"][icol]}) {ll.prod.DICT_STAT_LONGNAME[stat_name]}'
            ax.set_title(text)
    for ax in gsfig:
        ax.axis('off')
    ll.prod.savefig()
    plt.show()
    plt.close()
    return

if __name__=='__main__':
    #save(*sys.argv)
    draw(*sys.argv)