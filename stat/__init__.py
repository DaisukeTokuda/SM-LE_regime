import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import pdutil
import dictutil
import figlib
from util import stat as statutil

import lsmconst as ll

from . import bpdef

KEYS = ('path0', 'path1', 'var_name')
DICT_STAT_BEST = {
    'bic': 'min',
    'r2': 'max',
    'rmse': 'min',
    'Accuracy': 'max',
    'Precision': 'max',
    'Recall': 'max',
    'F1 score': 'max',
    'AUC': 'max',
}
DICT_STAT_FIXZERO = {
    'bic': False,
    'r2': True,
    'rmse': True,
    'Accuracy': 'max',
    'Precision': 'max',
    'Recall': 'max',
    'F1 score': 'max',
    'AUC': 'max',
}

# ===================================================================================================
def get_path(stat_name):
    '''
    Args:
        stat_name (str)
    Returns:
        (str)
    '''
    return os.path.join(ll.config.BS_DIR, 'out', 'stat0d', stat_name + '.csv')


def read(stat_name, log=True):
    '''
    Args:
        stat_name (str)
        log (bool)
    Returns:
        df (pandas.DataFrame)
    '''
    path = get_path(stat_name)
    df = pd.read_csv(path)
    if log: print(path, df.shape)
    return df

# ===================================================================================================
def get_columns(stat_name):
    columns = ['path0', 'path1', 'var_name', 'num0', 'num1', 'and01']
    if stat_name == 'corr':
        columns += ['corr', 'p']
    else:
        columns += [stat_name]
    return columns


def get_dtypes(stat_name):
    dtypes = ['str', 'str', 'str', 'int', 'int', 'int']
    if stat_name == 'corr':
        dtypes += ['float', 'float']
    else:
        dtypes += ['float']
    return dtypes

# ===================================================================================================
def get_values(catalog, attr_name, statdf, stat_column_name):
    '''
    Args:
        catalog (pd.DataFrame)
        attr_name (str)
        statdf (pd.DataFrame)
        stat_column_name (str)
    Returns:
        ret ([scalar,])
    '''
    ret = []

    attr_values = sorted(catalog[attr_name].unique())
    for attr_value0, attr_value1 in itertools.combinations(attr_values, 2):
        catalog0 = pdutil.filter(catalog, log=False, **{attr_name: attr_value0})
        catalog1 = pdutil.filter(catalog, log=False, **{attr_name: attr_value1})
        assert(len(catalog0) == 1)
        assert(len(catalog1) == 1)
        path0 = catalog0.iloc[0]['path']
        path1 = catalog1.iloc[0]['path']
        _statdf = pdutil.filter(statdf, log=False, path0=path0, path1=path1)
        if len(_statdf) == 0:
            _statdf = pdutil.filter(statdf, log=False, path0=path1, path1=path0)
        if not len(_statdf) == 1:
            print(attr_value0, attr_value1)
            continue
        _ret = _statdf.iloc[0][stat_column_name]
        if np.isnan(_ret): continue
        ret += [_ret]
    return ret

# ===================================================================================================
def dict2df(d, key_columns, value_column):
    data = {k: [] for k in key_columns}
    data[value_column] = []
    for k, v in d.items():
        for column, val in zip(key_columns, k):
            data[column] += [val]
        data[value_column] += [v]
    df = pd.DataFrame.from_dict(data)
    print(data.keys(), df.shape)
    return df

# ===================================================================================================
def bar_stat(df, stat_names, prop_names, model_names):
    '''
    Rows: stat_names
    Cols: prop_names
    '''
    width_ax, height_ax = 3, 3
    row_values = stat_names
    col_values = prop_names
    nrows = len(row_values)
    ncols = len(col_values)

    gsfig = figlib.GridSpecFig((width_ax * len(col_values), height_ax * len(row_values)), nrows, ncols, wspace=0.5, hspace=0.3)
    for irow, row_value in enumerate(row_values):
        for icol, col_value in enumerate(col_values):
            arrs, heights = [], []
            for model_name in model_names:
                _df = pdutil.filter(df, log=False,
                    stat_name=row_value, prop_name=col_value, model_name=model_name)
                arr = np.array(_df['value'].to_list())
                arrs += [arr]
                heights += [arr.mean()]
            color = ['grey'] * len(model_names)
            heights = np.array(heights)
            idx = getattr(heights, f'arg{DICT_STAT_BEST[row_value]}')()
            color[idx] = 'royalblue'
            fix_zero = False if row_value == 'bic' else True
            ax = gsfig[irow, icol]
            #print(col_value, row_value, model_name)
            #print(arrs)
            ll.figure.bars_with_std(ax, arrs,
                labels=[model_name.split('_')[-1] for model_name in model_names], fix_zero=fix_zero, color=color)
    for ax, col_value in zip(gsfig[0, :], col_values):
        ax.set_title(col_value)
    for ax, row_value in zip(gsfig[:, 0], row_values):
        ax.set_ylabel(row_value)

    return gsfig

# ===================================================================================================
def box_in_exp(df, stat_names, prop_names, model_names):
    col_values = prop_names
    row_values = stat_names
    width_ax, height_ax = 3, 3
    nrows = len(row_values)
    ncols = len(col_values)

    gsfig = figlib.GridSpecFig((width_ax * len(col_values), height_ax * len(row_values)), nrows, ncols, wspace=0.5, hspace=0.3)
    for irow, row_value in enumerate(row_values):
        for icol, col_value in enumerate(col_values):
            arrs = []
            for model_name in model_names:
                _df = pdutil.filter(df, log=False,
                    stat_name=row_value, prop_name=col_value, model_name=model_name)
                _arrs = []
                for source_id, __df in _df.groupby('source_id'):
                    val = np.array(__df['value'].to_list()).mean()
                    _arrs += [val]
                arrs += [_arrs]

            ax = gsfig[irow, icol]
            ax.boxplot(arrs,
                whis=(10, 90), showfliers=True,
                labels=[model_name.split('_')[-1] for model_name in model_names])
            if row_value in ['r2', 'rmse']:
                _, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax)
            texts = [f'n = {len(arrs[0])}']
            figlib.util.text_loc(ax, texts)
    for ax, col_value in zip(gsfig[0, :], col_values):
        ax.set_title(col_value)
    for ax, row_value in zip(gsfig[:, 0], row_values):
        ax.set_ylabel(row_value)
    return gsfig


def remove_sources_less_models(df, model_names):
    for source_id, _df in df.groupby('source_id'):
        if len(_df['model_name'].unique()) < len(model_names):
            df.drop(_df.index, inplace=True)
    return df


def remove_sources_less_experiments(df):
    for source_id, _df in df.groupby('source_id'):
        if len(_df['experiment_id'].unique()) < 2:
            df.drop(_df.index, inplace=True)
    return df