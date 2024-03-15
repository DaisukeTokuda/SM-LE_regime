#! /usr/bin/env python3
'''
Requirements:
    ./lasso_save.py
'''
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

import ioutil
import pdutil
from ncar import CMIP as cmip
from util import stat as statutil
from laiapy import csm
import figlib

import lsmconst as ll


model_type = ll.stat.bpdef.MODEL_TYPES[2]
ind_variable_ids = ['pr', 'rds']
file_name = model_type + '_' + '-'.join(ind_variable_ids)

prop_names = ll.PROP_NAMES[:2]
stat_names = ['f1', 'roc_auc']

# ===================================================================================================
def scatter_each_variable(ax, df):
    '''
    Row: stat_names
    Col: ofl_experiment_ids
    '''
    source_ids = ll.config.CPLOFL_SOURCE_IDS
    cmap = plt.get_cmap('jet')
    colors = [cmap(i / len(source_ids)) for i in range(len(source_ids))]

    cpls, ofls = [], []
    for source_id, color in zip(source_ids, colors):
        cpldf = pdutil.filter(df, log=False,
            source_id=source_id, experiment_id=ll.CPL_EXPERIMENT_IDS)
        #print(cpldf)
        arr = cpldf['value'].values
        cpl = arr.mean()
        cplerr = np.std(arr)
        if np.isnan(cplerr): cplerr = 0

        ofldf = pdutil.filter(df, log=False,
            source_id=source_id, experiment_id=ll.OFL_EXPERIMENT_IDS)
        #print(ofldf)
        #assert(len(__df) == 1)
        arr = ofldf['value'].values
        ofl = arr.mean()
        oflerr = np.std(arr)
        if np.isnan(oflerr): oflerr = 0

        #print(cplerr, oflerr)
        ax.errorbar(cpl, ofl, xerr=cplerr, yerr=oflerr,
            marker='o', linestyle='', color=color)

        cpls += [cpl]
        ofls += [ofl]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    vmin, vmax = min(xmin, ymin), max(xmax, ymax)

    corr, p = statutil.corr_with_p(np.array(cpls), np.array(ofls))
    texts = [f'CC = {corr:.3f}', f'p = {p:.3f}']
    figlib.util.text_loc(ax, texts, loc='lower right', prop={'size':10})

    figlib.util.scatter.plot_45line(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.hline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.vline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.set_xylim(ax, vmin, vmax)

    return


def exe_each_var(df):
    args = [
        (prop_names[0], ind_variable_ids[0], stat_names[0]),
        (prop_names[0], ind_variable_ids[1], stat_names[0]),
        (prop_names[1], ind_variable_ids[0], stat_names[0]),
        (prop_names[1], ind_variable_ids[1], stat_names[0]),
        (prop_names[0], ind_variable_ids[0], stat_names[1]),
        (prop_names[0], ind_variable_ids[1], stat_names[1]),
        (prop_names[1], ind_variable_ids[0], stat_names[1]),
        (prop_names[1], ind_variable_ids[1], stat_names[1]),
    ]

    #width_ax, height_ax = 3, 3
    width_ax, height_ax = 2.5, 2.5
    nrows, ncols = 2, 4
    gsfig = figlib.GridSpecFig(
        axsize=(width_ax, height_ax),
        nrows=nrows, ncols=ncols, wspace=0.4, hspace=0.3,
        direction='row',
    )
    for ax, (prop_name, variable_id, stat_name) in zip(gsfig, args):
        #ax.set_title(' '.join((prop_name, variable_id, stat_name)))
        _df = pdutil.filter(df,
            prop_name=prop_name, variable_id=variable_id, stat_name=stat_name, type='mean')
        scatter_each_variable(ax, _df)
        ax.text(0.02, 0.98, ll.figure.DICT_VARIABLE_SHORTNAME[variable_id],
            ha='left', va='top', transform=ax.transAxes, size=12)
        #break
    icol = 0
    for i, (ax, stat_name) in enumerate(zip(gsfig[:, icol], stat_names)):
        text = f'({["i", "ii"][i]}) ' + ll.stat.bpdef.DICT_SCORE_LONGNAME[stat_name]
        ax.set_ylabel(text, size=15)

    irow = 0
    ax = figlib.axesutil.merge_axes(gsfig[irow, 0], gsfig[irow, 1])
    ax.axis('off')
    text = '(a) ' + ll.figure.DICT_PROP_BPNAME[prop_names[0]]
    ax.text(0.5, 1.05, text,
        ha='center', va='bottom', transform=ax.transAxes, size=15)

    ax = figlib.axesutil.merge_axes(gsfig[irow, 2], gsfig[irow, 3])
    ax.axis('off')
    text = '(b) ' + ll.figure.DICT_PROP_BPNAME[prop_names[1]]
    ax.text(0.5, 1.05, text,
        ha='center', va='bottom', transform=ax.transAxes, size=15)

    irow = 1
    ax = figlib.axesutil.merge_axes(gsfig[irow, 0], gsfig[irow, 3])
    ax.axis('off')
    ax = figlib.axesutil.add_axes(ax, (0, -0.4, 1, 0.2))
    ax.axis('off')
    source_ids = ll.config.CPLOFL_SOURCE_IDS
    cmap = plt.get_cmap('jet')
    colors = [cmap(i / len(source_ids)) for i in range(len(source_ids))]
    figlib.util.legend(ax, [{'marker': 'o', 'color': color, 'ls': 'none'} for color in colors],
        labels=source_ids,
        #ncols=len(source_ids),
        ncols=4,
        loc='center', fontsize=12)

    ll.prod.savefig()
    plt.show()
    plt.close()

# ===================================================================================================
def scatter_variables(df):
    '''
    Columns: prop_names
    '''

    #width_ax, height_ax = 3, 3
    width_ax, height_ax = 2.5, 2.5

    nrows = len(stat_names)
    ncols = len(prop_names)

    gsfig = figlib.GridSpecFig(
        axsize=(width_ax, height_ax),
        nrows=nrows, ncols=ncols, wspace=0.5, hspace=0.3
    )

    for irow, stat_name in enumerate(stat_names):
        for icol, prop_name in enumerate(prop_names):
            cpldf = pdutil.filter(df, log=False,
                prop_name=prop_name, stat_name=stat_name, experiment_id=ll.CPL_EXPERIMENT_IDS)
            ofldf = pdutil.filter(df, log=False,
                prop_name=prop_name, stat_name=stat_name, experiment_id=ll.OFL_EXPERIMENT_IDS)

            ax = gsfig[irow, icol]
            ll.vs_atm.fig.scatter_variables(ax, cpldf, ofldf, ind_variable_ids, ll.config.OFL_EXPERIMENT_IDS)
            ax.text(0.98, 0, ll.figure.DICT_VARIABLE_SHORTNAME[ind_variable_ids[0]],
                ha='right', va='bottom', transform=ax.transAxes)
            ax.text(0.01, 0.99, ll.figure.DICT_VARIABLE_SHORTNAME[ind_variable_ids[1]],
                ha='left', va='top', transform=ax.transAxes)

            if gsfig.is_top(irow):
                text = f'({"abcd"[icol]}) ' + ll.figure.DICT_PROP_BPNAME[prop_name]
                text = ' '.join(text.split(' ')[:-1]) + '\n' + text.split(' ')[-1]
                ax.set_title(text)
            if gsfig.is_left(icol):
                text = f'({["i", "ii"][irow]}) ' + ll.stat.bpdef.DICT_SCORE_LONGNAME[stat_name]
                ax.set_ylabel(text)
    icol = -1
    ax = figlib.axesutil.merge_axes(gsfig[0, icol], gsfig[-1, icol])
    ax.axis('off')
    #ax = figlib.axesutil.add_axes(ax, (0, -0.3, 1, 0.2))
    ax = figlib.axesutil.add_axes(ax, (1, 0, 0.5, 1))
    ax.axis('off')
    source_ids = ll.config.CPLOFL_SOURCE_IDS
    cmap = plt.get_cmap('jet')
    colors = [cmap(i / len(source_ids)) for i in range(len(source_ids))]
    figlib.util.legend(ax, [{'marker': 'o', 'color': color, 'ls': 'none'} for color in colors],
        labels=source_ids, ncols=1,
        #loc='center',
        loc='lower left',
    )

    ll.prod.savefig('vars')
    plt.show()
    plt.close()



# ===================================================================================================
def main(*args):
    df = ll.vs_atm.read_importance(model_type, ind_variable_ids)
    exe_each_var(df)
    return
    for variable_id in ind_variable_ids:
        df[variable_id] = np.nan
        mask = (df['variable_id'] == variable_id) & (df['type'] == 'mean')
        df[variable_id][mask] = df[mask]['value']
    df = df[df['type'] == 'mean']
    scatter_variables(df)
    return

if __name__=='__main__':
    main(*sys.argv)