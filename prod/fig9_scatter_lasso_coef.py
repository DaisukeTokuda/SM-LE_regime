#! /usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import itertools

import ioutil
from ncar import CMIP as cmip
import figlib
from util import stat as statutil
import dictutil
import pdutil

import lsmconst as ll

model_type = 'lasso'
ind_variable_ids = ['pr', 'rds']

key_columns = ['source_id', 'experiment_id', 'member_id', 'prop_name', 'class_id']

prop_names = ll.PROP_NAMES

# ===================================================================================================
def scatter_each_variable(df):
    '''
    Row: ofl_experiment_ids
    Col: prop_names
    '''
    #width_ax, height_ax = 3, 3
    width_ax, height_ax = 2.5, 2.5
    args = []
    for prop_name in prop_names:
        for variable_id in ind_variable_ids:
            args += [(prop_name, variable_id)]

    nrows, ncols = 2, 4

    gsfig = figlib.GridSpecFig(
        axsize=(width_ax, height_ax), nrows=nrows, ncols=ncols,
        wspace=0.5, hspace=0.4,
    )

    for ax, (prop_name, variable_id) in zip(gsfig, args):
        cpldf = pdutil.filter(df, log=False,
            prop_name=prop_name, experiment_id=ll.CPL_EXPERIMENT_IDS)
        ofldf = pdutil.filter(df, log=False,
            prop_name=prop_name, experiment_id=ll.OFL_EXPERIMENT_IDS)
        ll.vs_atm.fig.scatter_each_variable(ax, cpldf, ofldf, variable_id)
        ax.text(0.02, 0.98, ll.figure.DICT_VARIABLE_SHORTNAME[variable_id],
            ha='left', va='top', transform=ax.transAxes, size=12)

    irow = 0
    ax = figlib.axesutil.merge_axes(gsfig[irow, 0], gsfig[irow, 1])
    ax.axis('off')
    text = '(a) ' + ll.figure.DICT_PROP_NAME[prop_names[0]]
    ax.text(0.5, 1.05, text,
        ha='center', va='bottom', transform=ax.transAxes, size=12)

    ax = figlib.axesutil.merge_axes(gsfig[irow, 2], gsfig[irow, 3])
    ax.axis('off')
    text = '(b) ' + ll.figure.DICT_PROP_NAME[prop_names[1]]
    ax.text(0.5, 1.05, text,
        ha='center', va='bottom', transform=ax.transAxes, size=12)

    irow = 1
    ax = figlib.axesutil.merge_axes(gsfig[irow, 0], gsfig[irow, 1])
    ax.axis('off')
    text = '(c) ' + ll.figure.DICT_PROP_NAME[prop_names[2]]
    ax.text(0.5, 1.05, text,
        ha='center', va='bottom', transform=ax.transAxes, size=12)

    ax = figlib.axesutil.merge_axes(gsfig[irow, 2], gsfig[irow, 3])
    ax.axis('off')
    text = '(d) ' + ll.figure.DICT_PROP_NAME[prop_names[3]]
    ax.text(0.5, 1.05, text,
        ha='center', va='bottom', transform=ax.transAxes, size=12)

    irow = 1
    ax = figlib.axesutil.merge_axes(gsfig[irow, 0], gsfig[irow, 3])
    ax.axis('off')
    ax = figlib.axesutil.add_axes(ax, (0, -0.45, 1, 0.2))
    ax.axis('off')
    source_ids = ll.config.CPLOFL_SOURCE_IDS
    cmap = plt.get_cmap('jet')
    colors = [cmap(i / len(source_ids)) for i in range(len(source_ids))]
    figlib.util.legend(ax, [{'marker': 'o', 'color': color, 'ls': 'none'} for color in colors],
        labels=source_ids,
        #ncols=len(source_ids),
        ncols=4,
        loc='center',
        fontsize=12)

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

    nrows = 1
    ncols = len(prop_names)

    gsfig = figlib.GridSpecFig(
        (width_ax * ncols, height_ax * nrows),
        nrows, ncols, wspace=0.3, hspace=0.3
    )

    for ax, prop_name in zip(gsfig, prop_names):
        cpldf = pdutil.filter(df, log=False,
            prop_name=prop_name, experiment_id=ll.CPL_EXPERIMENT_IDS)
        ofldf = pdutil.filter(df, log=False,
            prop_name=prop_name, experiment_id=ll.OFL_EXPERIMENT_IDS)
        ll.vs_atm.fig.scatter_variables(ax, cpldf, ofldf, ind_variable_ids, ll.config.OFL_EXPERIMENT_IDS)
        ax.text(0.98, 0, ll.figure.DICT_VARIABLE_SHORTNAME[ind_variable_ids[0]],
            ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.01, 0.99, ll.figure.DICT_VARIABLE_SHORTNAME[ind_variable_ids[1]],
            ha='left', va='top', transform=ax.transAxes)

    for icol, (ax, prop_name) in enumerate(zip(gsfig, prop_names)):
        text = f'({"abcd"[icol]}) ' + ll.figure.DICT_PROP_NAME[prop_name]
        ax.set_title(text)

    irow = 0
    ax = figlib.axesutil.merge_axes(gsfig[irow, 0], gsfig[irow, 3])
    ax.axis('off')
    ax = figlib.axesutil.add_axes(ax, (0, -0.3, 1, 0.2))
    ax.axis('off')
    source_ids = ll.config.CPLOFL_SOURCE_IDS
    cmap = plt.get_cmap('jet')
    colors = [cmap(i / len(source_ids)) for i in range(len(source_ids))]
    figlib.util.legend(ax, [{'marker': 'o', 'color': color, 'ls': 'none'} for color in colors],
        labels=source_ids, ncols=len(source_ids), loc='center')

    ll.prod.savefig('vars')
    plt.show()
    plt.close()

# ===================================================================================================
def main(*args):
    df = ll.vs_atm.read_importance(model_type, ind_variable_ids)
    df = pdutil.filter(df, log=False,
        source_id='~GISS-E2-1-G')
    for ind_variable_id in ind_variable_ids:
        df = df[df[ind_variable_id] > -1e3]
    print(df.shape)

    scatter_each_variable(df)
    #scatter_variables(df)
# ===================================================================================================
if __name__=='__main__':
    main(*sys.argv)
