#! /usr/bin/env python3
import os 
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from ncar import CMIP as cmip
import figlib

import lsmconst as ll

dir_name = 'prop_enstat_monthly'
cond = {
    'source_id': ll.CPL_SOURCE_IDS,
    'member_id': '*members',
    'grid_label': '2x2',
    'time_range': '190301-201212',
}

# ===================================================================================================
def mask_less_sample(arr, sample_frac, min_frac):
    mask = np.ma.masked_less(sample_frac, min_frac)
    return np.ma.masked_array(arr, mask=mask.mask)


def save(*args):
    ens_attr_name = 'member_id'
    loop_attr_name = 'source_id'

    min_member_frac = 0.5

    catalog = ll.catalog.read(dir_name)
    catalog = cmip.catalog.filter(catalog, **cond)
    #min_members = 10
    #catalog = ll.catalog.filter_member_num(catalog, ens_attr_name, min_members)

    attr_values = sorted(list(catalog[loop_attr_name].unique()))
    print(len(attr_values), attr_values)

    data = {prop_name: {} for prop_name in ll.figure.PROP_NAMES}
    data_multimodel = {prop_name: [] for prop_name in ll.figure.PROP_NAMES}
    for attr_value in attr_values:
        _catalog = cmip.catalog.filter(catalog, log=False, **{loop_attr_name: attr_value})
        assert(len(_catalog) == 1)
        path = _catalog.iloc[0]['path']
        member_num = ll.file.member2int(_catalog.iloc[0][ens_attr_name])
        div_member_num = 1.0 / float(member_num)
        print(path)
        ds = xr.open_dataset(path)

        for prop_name in ll.figure.PROP_NAMES:
            arr = ds[prop_name + '_std'].to_masked_array()
            sample_frac = ds[prop_name + '_num'].to_masked_array() * div_member_num
            arr = mask_less_sample(arr, sample_frac, min_member_frac)

            data[prop_name][attr_value] = arr[~arr.mask]

            var_name = prop_name + '_mean'
            arr = ds[var_name].to_masked_array()
            data_multimodel[prop_name] += [arr]

    for prop_name in ll.PROP_NAMES:
        arr = data_multimodel[prop_name]
        arr = np.ma.stack(arr, axis=0)
        std = np.ma.std(arr, axis=0)
        num = (~arr.mask).sum(axis=0)
        sample_frac = num * (1.0 / len(arr))
        arr = mask_less_sample(std, sample_frac, min_member_frac)
        data[prop_name]['Inter-Model'] = arr[~arr.mask]

    ll.prod.dump_pickle(data)
    return

# ---------------------------------------------------------------------------------------------------
def draw(*args):
    data = ll.prod.load_pickle()
    nrows = len(ll.figure.PROP_NAMES)
    ncols = 1
    figsize = (10, 10)
    source_ids = ll.config.CPL_SOURCE_IDS
    gsfig = figlib.GridSpecFig(figsize, nrows, ncols, hspace=0.2)

    for irow, (prop_name, dict_src_arr) in enumerate(data.items()):
        ax = gsfig[irow, 0]
        arrs = [dict_src_arr[k] for k in source_ids]
        arrs += [[]]
        arrs += [dict_src_arr['Inter-Model']]
        labels = ll.config.CPL_SOURCE_IDS + [''] + ['Inter-Model']
        ax.boxplot(arrs, labels=labels, whis=(5, 95),
            showfliers=False)
        if gsfig.is_bottom(irow):
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        else:
            ax.set_xticklabels([])

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
    l = ax.plot([xmin + 0.1, len(source_ids) + 1], [0.08, 0.08],
        ls='-', marker='|', color='k', lw=1, zorder=99)
    l[0].set_clip_on(False)
    l = ax.text(0.5 * (xmin + len(source_ids) + 1), 0.081, 'Intra-Model',
        ha='center', va='bottom', zorder=99)
    l[0].set_clip_on(False)

    ll.prod.savefig()
    plt.show()
    plt.close()
    return

if __name__=='__main__':
    #save(*sys.argv)
    draw(*sys.argv)
