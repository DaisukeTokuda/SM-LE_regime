#! /usr/bin/env python3
import os 
import sys
import numpy as np
import xarray as xr

from ncar import CMIP as cmip
import pdutil

import lsmconst as ll

# ===================================================================================================
def mask_less_sample(arr, sample_frac, min_frac):
    mask = np.ma.masked_less(sample_frac, min_frac)
    return np.ma.masked_array(arr, mask=mask.mask)


def calc_std(catalog, min_frac):
    '''
    Args:
        catalog (pd.DataFrame)
        min_frac (scalar): 0 - 1
    Returns:
        dict_var_std ({str: np.ma.MaskedArray})
    '''
    dict_var_arr = {var_name: [] for var_name in ll.figure.PROP_NAMES}
    for _, sr in catalog.iterrows():
        path = sr['path']
        ds = xr.open_dataset(path)

        for var_name in dict_var_arr.keys():
            arr = ds[var_name].to_masked_array()
            dict_var_arr[var_name] += [arr]
    
    dict_var_std = {}
    for var_name in dict_var_arr.keys():
        arr = np.ma.stack(dict_var_arr[var_name], axis=0)
        sample_size = (~arr.mask).sum(axis=0)
        mask = (sample_size / len(arr)) < min_frac
        std = np.ma.std(arr, axis=0)
        std = np.ma.masked_array(std, mask=mask)
        dict_var_std[var_name] = std[~std.mask].flatten()
    return dict_var_std

# ---------------------------------------------------------------------------------------------------
def save(*args):
    dir_name = 'prop_monthly'
    source_ids = ll.OFL_SOURCE_IDS
    cond = {
        'experiment_id': ll.OFL_EXPERIMENT_IDS,
        'source_id': source_ids,
        'grid_label': '2x2',
        'time_range': '190301-201212',
    }

    min_member_frac = 0.5

    catalog = ll.catalog.read(dir_name)
    catalog = cmip.catalog.filter(catalog, **cond)

    data = {} # (prop_name, source_id): 
    # same model, different forcing
    for source_id in source_ids:
        _catalog = pdutil.filter(catalog, source_id=source_id)
        if len(_catalog) < 2: continue
        dict_var_std = calc_std(_catalog, min_member_frac)
        for prop_name, arr in dict_var_std.items():
            data[(prop_name, source_id)] = arr

    for experiment_id in ll.OFL_EXPERIMENT_IDS:
        _catalog = pdutil.filter(catalog, experiment_id=experiment_id)
        if len(_catalog) < 2: continue
        dict_var_std = calc_std(_catalog, min_member_frac)
        for prop_name, arr in dict_var_std.items():
            data[(prop_name, experiment_id)] = arr

    ll.prod.dump_pickle(data)
    return

# ---------------------------------------------------------------------------------------------------
def draw(*args):
    import matplotlib.pyplot as plt

    import figlib

    data = ll.prod.load_pickle()
    nrows = len(ll.PROP_NAMES)
    ncols = 1
    figsize = (10, 10)
    gsfig = figlib.GridSpecFig(figsize, nrows, ncols, hspace=0.2)

    for irow, prop_name in enumerate(ll.PROP_NAMES):
        arrs, labels = [], []
        for source_id in ll.OFL_SOURCE_IDS:
            arrs += [data[(prop_name, source_id)]]
            labels += [source_id]
        arrs += [[]]
        labels += ['']
        for experiment_id in ll.OFL_EXPERIMENT_IDS:
            arrs += [data[(prop_name, experiment_id)]]
            labels += [experiment_id]
        ax = gsfig[irow, 0]
        ax.boxplot(arrs, labels=labels,
            whis=(5, 95), showfliers=False)
        if gsfig.is_bottom(irow):
            for tick in ax.get_xticklabels(): tick.set_rotation(90)
        else:
            ax.set_xticks([])

        text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAMEUNIT[prop_name]}'
        ax.text(0.005, 0.98, text,
            ha='left', va='top', transform=ax.transAxes)

        vmin, vmax = ll.prod.DICT_STD_RANGE[prop_name]
        ax.set_ylim(vmin, vmax)


    ax = gsfig[0, 0]
    #ax = gsfig[3, 0]
    xmin, xmax = ax.get_xlim()
    #ax = figlib.axesutil.add_axes(ax, (0, -0.8, 1, 0.1))
    #ax.axis('off')
    #ax.set_xlim(xmin, xmax)
    ax.plot([xmin + 0.1, len(ll.OFL_SOURCE_IDS) + 1, xmax - 0.1], [0.08, 0.08, 0.08],
        ls='-', marker='|', color='k', lw=1, zorder=99)
    ax.text(0.5 * (xmin + len(ll.OFL_SOURCE_IDS) + 1), 0.081, 'Intra-Model',
        ha='center', va='bottom', zorder=99)
    ax.text(0.5 * (xmax + len(ll.OFL_SOURCE_IDS) + 1), 0.081, 'Inter-Model',
        ha='center', va='bottom', zorder=99)
    ll.prod.savefig()
    plt.show()
    plt.close()
    return

if __name__=='__main__':
    save(*sys.argv)
    draw(*sys.argv)
