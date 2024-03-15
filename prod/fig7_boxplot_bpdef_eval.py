#! /usr/bin/env python3
'''
Requirements:
    ./lasso_save.py
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

import ioutil
import pdutil
import figlib

import lsmconst as ll

prop_names = ll.PROP_NAMES[:2]

model_type = ll.stat.bpdef.MODEL_TYPES[2]
ind_variable_ids = ['pr', 'rds']

#stat_names = ['Accuracy', 'Precision', 'Recall']
stat_names = ['f1', 'roc_auc']
dict_stat_range = {
    'f1': (0, 1),
    'roc_auc': (0.6, 1),
}

cmap = plt.get_cmap('rainbow')
n = 4
colors = [cmap(i / n) for i in range(n)]

# ===================================================================================================
def boxplot_groups(ax, dict_group_arrs):
    interval = 0.3 # between groups
    #buffer = 0.2 # from left and right edges

    for i, arrs in enumerate(dict_group_arrs.values()):
        width = (1 - interval) / len(arrs)
        _x = i - width * 0.5 * len(arrs)

        for arr, color in zip(arrs, colors):
            bp = ax.boxplot(arr, positions=[_x],
                whis=(10, 90), showfliers=False, widths=width)
            _x += width
            for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[item], color=color)
            #plt.setp(bp['medians'], color=color)

    ax.set_xticks(range(len(dict_group_arrs)))
    ax.set_xticklabels(dict_group_arrs.keys())
    return

# ===================================================================================================
def draw(ax, df):
    #df = ll.stat.remove_sources_less_experiments(df)
    source_ids = ll.config.CPLOFL_SOURCE_IDS
    print(len(source_ids), source_ids)

    c2c, o2o, c2o, o2c = [], [], [], []
    for source_id in source_ids:
        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
        )
        c2c += [_df['value'].values]

        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
        )
        o2o += [_df['value'].values]

        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
        )
        c2o += [_df['value'].values]

        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
        )
        o2c += [_df['value'].values]
    dict_data = {'C2C': c2c, 'O2O': o2o, 'C2O': c2o, 'O2C': o2c}
    boxplot_groups(ax, dict_data)
    return

# ===================================================================================================
def draw2(ax, df):
    #df = ll.stat.remove_sources_less_experiments(df)
    source_ids = ll.config.CPLOFL_SOURCE_IDS
    print(len(source_ids), source_ids)

    dict_source_data = {}
    for source_id in source_ids:
        dict_source_data[source_id] = []
        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
        )
        dict_source_data[source_id] += [_df['value'].values]

        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
        )
        dict_source_data[source_id] += [_df['value'].values]

        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
        )
        dict_source_data[source_id] += [_df['value'].values]

        _df = pdutil.filter(df,
            model_source_id=source_id, model_experiment_id=ll.config.OFL_EXPERIMENT_IDS,
            target_source_id=source_id, target_experiment_id=ll.config.CPL_EXPERIMENT_IDS,
        )
        dict_source_data[source_id] += [_df['value'].values]
    boxplot_groups(ax, dict_source_data)
    return

# ===================================================================================================
def main(*args):
    dir_name = ll.vs_atm.get_dir_name(model_type, ind_variable_ids)
    src_path = ll.vs_atm.get_score_path(dir_name)
    df = pd.read_csv(src_path)
    print(dir_name, df.shape)

    plt.rcParams["font.size"] = 14

    width_ax, height_ax = 5, 2
    nrows = len(stat_names)
    ncols = len(prop_names)
    gsfig = figlib.GridSpecFig(nrows=nrows, ncols=ncols, axsize=(width_ax, height_ax), wspace=0.1, hspace=0.1)

    #bar_each_sim(df)
    for (irow, stat_name), (icol, prop_name) in itertools.product(enumerate(stat_names), enumerate(prop_names)):
        cond = {
            'prop_name': prop_name,
            'stat_name': stat_name,
            'model_source_id': ll.config.CPLOFL_SOURCE_IDS,
            'target_source_id': ll.config.CPLOFL_SOURCE_IDS,
        }
        _df = pdutil.filter(df, **cond)

        ax = gsfig[irow, icol]
        #draw(ax, _df)
        draw2(ax, _df)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin + 0.3, xmax - 0.3)

        ax.set_ylim(dict_stat_range[stat_name])
        if not gsfig.is_left(icol): ax.set_yticklabels([])
        if not gsfig.is_bottom(irow):
            ax.set_xticklabels([])
        else:
            for tick in ax.get_xticklabels(): tick.set_rotation(90)
        #break

    irow = 0
    for icol, prop_name in enumerate(prop_names):
        ax = gsfig[irow, icol]
        text = f'({"ab"[icol]}) {ll.figure.DICT_PROP_BPNAME[prop_name]}'
        ax.set_title(text)

    icol = 0
    for irow, stat_name in enumerate(stat_names):
        ax = gsfig[irow, icol]
        text = f'({["i", "ii"][irow]}) {ll.stat.bpdef.DICT_SCORE_LONGNAME[stat_name]}'
        ax.set_ylabel(text)



    irow = -1
    ax = gsfig.get_merged_axes(irow=irow)
    ax.axis('off')
    ax = figlib.axesutil.add_axes(ax, (0, -1.4, 1, 0.2))
    ax.axis('off')
    labels = ['C2C', 'O2O', 'C2O', 'O2C']
    figlib.util.legend(ax, [{'marker': 'none', 'color': color, 'ls': '-'} for color in colors],
        labels=labels, ncols=4, loc='lower center')

    ll.prod.savefig('2')
    plt.show()
    plt.close()

    return

if __name__=='__main__':
    main(*sys.argv)