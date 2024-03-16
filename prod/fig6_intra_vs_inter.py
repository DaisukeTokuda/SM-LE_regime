#! /usr/bin/env python3
import os 
import sys
import numpy as np
import itertools
import xarray as xr
import matplotlib.pyplot as plt
import itertools

import figlib
from util import stat as statutil
from ncar import CMIP as cmip

import lsmconst as ll

stat_names = ['JC', 'CC', 'RMSD']

# ===================================================================================================
def calc_stat(cpl_sr, ofl_sr, min_frac):
    '''
    Args:
        cpl_sr (pd.Series)
        ofl_sr (pd.Series)
        min_frac (scalar): 0 - 1
    Returns:
        ret ({str: scalar}): {'JC'; *, 'CC': *, 'RMSD': *}
    '''
    cpl_path = cpl_sr['path']
    ofl_path = ofl_sr['path']
    cpl_ds = xr.open_dataset(cpl_path)
    ofl_ds = xr.open_dataset(ofl_path)

    ret = {}
    for prop_name in ll.figure.PROP_NAMES:
        cpl_arr = cpl_ds[prop_name + '_mean'].to_masked_array().flatten()
        cpl_num = cpl_ds[prop_name + '_num'].to_masked_array().flatten()
        cpl_mask = (cpl_num / ll.catalog.get_member_num(cpl_sr['member_id'])) < min_frac
        cpl_arr = np.ma.masked_array(cpl_arr, mask=cpl_mask)
        ofl_arr = ofl_ds[prop_name + '_mean'].to_masked_array().flatten()

        ret[(prop_name, 'JC')] = statutil.jaccard(cpl_arr, ofl_arr)
        cpl_arr, ofl_arr = statutil.filter_invalid(cpl_arr, ofl_arr)
        ret[(prop_name, 'CC')] = statutil.corr(cpl_arr, ofl_arr)
        ret[(prop_name, 'RMSD')] = statutil.rmsd(cpl_arr, ofl_arr)
    return ret


def save(*args):
    dir_name = 'prop_enstat_monthly'
    cpl_cond = {
        'source_id': ll.CPLOFL_SOURCE_IDS,
        'member_id': '*members',
        'grid_label': '2x2',
        'time_range': '190301-201212',
    }
    min_frac = 0.5

    ofl_cond = {
        #'source_id': '~GISS-E2-1-G',
        'source_id': ll.CPLOFL_SOURCE_IDS,
        'experiment_id': '*members',
        'grid_label': '2x2',
        'time_range': '190301-201212',
    }

    catalog = ll.catalog.read(dir_name)
    cpl_catalog = cmip.catalog.filter(catalog, **cpl_cond)
    ofl_catalog = cmip.catalog.filter(catalog, **ofl_cond)
    ofl_catalog = ll.catalog.filter_single_member(ofl_catalog)

    data = {}
    for prop_name, stat_name in itertools.product(ll.PROP_NAMES, stat_names):
        data[(prop_name, stat_name, 'intra')] = []
        data[(prop_name, stat_name, 'inter')] = []

    for source_id in ll.CPLOFL_SOURCE_IDS:
        _cpl_catalog = cmip.catalog.filter(cpl_catalog, log=False, source_id=source_id)
        assert(len(_cpl_catalog) == 1)
        _cpl_sr = _cpl_catalog.iloc[0]

        for _source_id in ll.CPLOFL_SOURCE_IDS:
            _ofl_catalog = cmip.catalog.filter(ofl_catalog, log=False, source_id=_source_id)
            assert(len(_ofl_catalog) == 1)
            _ofl_sr = _ofl_catalog.iloc[0]

            dict_stat = calc_stat(_cpl_sr, _ofl_sr, min_frac)
            mode = 'intra' if source_id == _source_id else 'inter'
            for prop_name, stat_name in itertools.product(ll.PROP_NAMES, stat_names):
                data[(prop_name, stat_name, mode)] += [dict_stat[(prop_name, stat_name)]]

    ll.prod.dump_pickle(data)
    return

# ===================================================================================================
def draw(*args):

    width_ax, height_ax = 2, 1.5
    nrows, ncols = 4, len(stat_names)
    dict_stat_lims = {
        #'JC': (0.1, 0.7),
        #'CC': (0.4, 1),
    }

    axsize = (width_ax, height_ax * 1.2)
    gsfig = figlib.GridSpecFig(nrows=nrows, ncols=ncols, axsize=axsize,
        wspace=0.4, hspace=0.2)

    data = ll.prod.load_pickle()
    for (irow, prop_name), (icol, stat_name) in itertools.product(enumerate(ll.PROP_NAMES), enumerate(stat_names)):
        ax = gsfig[irow, icol]
        _data = [data[(prop_name, stat_name, mode)] for mode in ('intra', 'inter')]
        bp = ax.boxplot(_data, showfliers=False,
            whis=(5, 95))
        plt.setp(bp['medians'][1], color='royalblue')
        print(ax.get_xlim())
        ax.set_xlim(0.8, 2.2)
        if not gsfig.is_bottom(irow): ax.set_xticks([])
        if stat_name in dict_stat_lims:
            vmin, vmax = dict_stat_lims[stat_name]
            ax.set_ylim(vmin, vmax)

    if False:
        # source_id
        irow = 0
        for icol, stat_name in enumerate(stat_names):
            text = f' ({("i", "ii", "iii")[icol]}) {stat_name}'
            if icol == 0:
                text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAMEUNIT[ll.PROP_NAMES[irow]]}\n' + text
            ax = gsfig[irow, icol]
            ax.text(-0.02, 1.05, text, va='bottom', ha='left',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes)

        # prop_names
        icol = 0
        for irow, prop_name in enumerate(ll.PROP_NAMES[1:], start=1):
            text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAMEUNIT[prop_name]}'
            ax = gsfig[irow, icol]
            ax.text(-0.02, 1.05, text, va='bottom', ha='left',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes)

    irow = 0
    for icol, stat_name in enumerate(stat_names):
        text = f' ({("i", "ii", "iii")[icol]}) {stat_name}'
        ax = gsfig[irow, icol]
        ax.set_title(text)

    irow = -1
    for ax in gsfig[irow, :]:
        #ax.set_xticks([0, 1])
        ax.set_xticklabels(['Intra-model', 'Inter-model'])

    icol = 0
    for irow, prop_name in enumerate(ll.PROP_NAMES):
        text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAME[prop_name]}'
        ax = gsfig[irow, icol]
        ax.set_ylabel(text)

    ll.prod.savefig()
    plt.show()
    plt.close()
    return

def draw_kernel_density(*args):
    from sklearn.neighbors import KernelDensity
    stat_names = ['JC', 'CC', 'RMSD']
    width_ax, height_ax = 2, 1.5
    nrows, ncols = 4, len(stat_names)

    data = ll.prod.load_pickle()
    axsize = (1, 1)
    gsfig = figlib.GridSpecFig(nrows=nrows, ncols=ncols, axsize=axsize,
        wspace=0.4, hspace=0.2)
    dict_range = {}
    for (irow, prop_name), (icol, stat_name) in itertools.product(enumerate(ll.PROP_NAMES), enumerate(stat_names)):
        ax = gsfig[irow, icol]
        _data = [data[(prop_name, stat_name, mode)] for mode in ('intra', 'inter')]
        bp = ax.boxplot(_data, showfliers=False,
            whis=(5, 95))
        vmin, vmax = ax.get_ylim()
        dict_range[(prop_name, stat_name)] = (vmin, vmax)
    del gsfig

    # -----------------------------------------------------------------------------------------------

    axsize = (2, 1.5)
    gsfig = figlib.GridSpecFig(nrows=nrows, ncols=ncols, axsize=axsize,
        wspace=0.4, hspace=0.3)
    kernel = 'gaussian'
    #kernel = 'epanechnikov'
    dict_mode_color = {
        'intra': 'royalblue',
        'inter': 'crimson',
    }
    dict_mode_longname = {
        'intra': 'Intra-model',
        'inter': 'Inter-model',
    }
    for (irow, prop_name), (icol, stat_name) in itertools.product(enumerate(ll.PROP_NAMES), enumerate(stat_names)):
        print(prop_name, stat_name)
        ax = gsfig[irow, icol]

        vmin, vmax = dict_range[(prop_name, stat_name)]
        vdif = vmax - vmin
        vmin -= 0.8 * vdif
        vmax += 0.8 * vdif
        if stat_name in ('JC', 'CC'): vmax = min(vmax, 1)
        if stat_name == 'RMSD': vmin = max(vmin, 0)
        #else:
        #    vmin, vmax = 0, 0.15
        x = np.linspace(vmin, vmax, 1000)

        for mode in ('intra', 'inter'):
            _data = np.array(data[(prop_name, stat_name, mode)])
            bandwidth = scott=np.sqrt(np.var(_data, ddof=1)*((len(_data))**(-1/5))**2) # Scott
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(_data[:, None])
            log_dens = kde.score_samples(x[:, None])
            ax.plot(x, np.exp(log_dens), color=dict_mode_color[mode], label=dict_mode_longname)
        ax.set_xlim(vmin, vmax)

        if gsfig.is_bottom(irow) and gsfig.is_right(icol):
            _ax = figlib.axesutil.add_axes(ax, (0, -0.4, 1, 0.2))
            figlib.util.legend(_ax,
                [{'color': dict_mode_color[mode]} for mode in ('intra', 'inter')],
                [dict_mode_longname[mode] for mode in ('intra', 'inter')],
                ncols=2)

    irow = 0
    for icol, stat_name in enumerate(stat_names):
        text = f' ({("i", "ii", "iii")[icol]}) {stat_name}'
        ax = gsfig[irow, icol]
        ax.set_title(text)

    icol = 0
    for irow, prop_name in enumerate(ll.PROP_NAMES):
        text = f'({"abcd"[irow]}) {ll.figure.DICT_PROP_NAME[prop_name]}'
        ax = gsfig[irow, icol]
        ax.set_ylabel(text)

    ll.prod.savefig()
    plt.show()
    plt.close()
# ===================================================================================================
if __name__=='__main__':
    #save(*sys.argv)
    #draw(*sys.argv)
    draw_kernel_density(*sys.argv)