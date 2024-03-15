import numpy as np
import matplotlib.pyplot as plt

import pdutil
import figlib
from util import stat as statutil
   
def scatter_each_variable(ax, cpldf, ofldf, value_column_name):
    '''
    Row: stat_names
    Col: ofl_experiment_ids
    Axes:
        x: Coupled (ensemble mean) + error bar (ensemble STD)
        y: Offline
    '''
    cmap = plt.get_cmap('jet')
    source_ids = list(ofldf['source_id'].unique())
    colors = [cmap(i / len(source_ids)) for i in range(len(source_ids))]

    cpl, cplerr, ofl, oflerr = [], [], [], []
    for source_id, color in zip(source_ids, colors):
        _cpldf = pdutil.filter(cpldf, log=False,
            source_id=source_id)
        arr = _cpldf[value_column_name].values
        cpl += [arr.mean()]
        cplerr += [np.std(arr)]

        _ofldf = pdutil.filter(ofldf, log=False,
            source_id=source_id)
        print(source_id, _cpldf.shape, _ofldf.shape)
        #assert(len(_ofldf) == 1)
        arr = _ofldf[value_column_name].values
        ofl += [arr.mean()]
        oflerr += [np.std(arr)]

        ax.errorbar(cpl[-1], ofl[-1], xerr=cplerr[-1], yerr=oflerr[-1], color=color,
            marker='o', linestyle='', label=source_id)

    #ax.errorbar(cpl, ofl, xerr=cplerr, color=colors,
    #    marker='o', linestyle='', label=source_ids)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    vmin, vmax = min(xmin, ymin), max(xmax, ymax)

    try:
        corr, p = statutil.corr_with_p(np.array(cpl), np.array(ofl))
    except ValueError: # less sample size
        pass
    else:
        texts = [f'CC = {corr:.3f}', f'p = {p:.3f}']
        figlib.util.text_loc(ax, texts, loc='lower right', prop={'size':10})

    figlib.util.scatter.plot_45line(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.hline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.vline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.set_xylim(ax, vmin, vmax)
    return ax


def scatter_variables(ax, cpldf, ofldf, ind_variable_ids, ofl_experiment_ids):
    keys = ['x', 'y']
    cmap = plt.get_cmap('jet')
    source_ids = list(cpldf['source_id'].unique())
    colors = [cmap(i / len(source_ids)) for i in range(len(source_ids))]

    for source_id, color in zip(source_ids, colors):
        _cpldf = pdutil.filter(cpldf, log=False, source_id=source_id)
        _ofldf = pdutil.filter(ofldf, log=False, source_id=source_id)
        if len(_cpldf) == 0 or len(_ofldf) == 0: continue

        dict_cpl_mean, dict_cpl_std = {}, {}
        for key, variable_id in zip(keys, ind_variable_ids):
            arr = _cpldf[variable_id].values
            arr = arr[~np.isnan(arr)]
            dict_cpl_mean[key] = arr.mean()
            dict_cpl_std[key] = np.std(arr)

        dict_ofl_mean = {key: [] for key in keys}
        for experiment_id in ofl_experiment_ids:
            __ofldf = pdutil.filter(_ofldf, log=False, experiment_id=experiment_id)
            if len(__ofldf) == 0:
                for key in keys: dict_ofl_mean[key] += [None]
                continue
            #print(source_id, experiment_id)
            #print(__ofldf[variable_id])
            for key, variable_id in zip(keys, ind_variable_ids):
                arr = __ofldf[variable_id].values
                arr = arr[~np.isnan(arr)]
                dict_ofl_mean[key] += [arr[0]]
        if all([v is None for v in dict_ofl_mean['x']]): continue

        for i in range(len(ofl_experiment_ids)):
            x = [dict_cpl_mean['x'], dict_ofl_mean['x'][i]]
            y = [dict_cpl_mean['y'], dict_ofl_mean['y'][i]]
            if x[-1] is None: continue
            xerr = [dict_cpl_std['x'], 0]
            yerr = [dict_cpl_std['y'], 0]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                marker='', alpha=0.3,
                color=color)
        ax.scatter(dict_cpl_mean['x'], dict_cpl_mean['y'],
            marker='o', color=color,
            label=source_id)
        ax.scatter(dict_ofl_mean['x'], dict_ofl_mean['y'],
            marker='o', edgecolors=color, facecolor='None')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    vmin, vmax = min(xmin, ymin), max(xmax, ymax)

    figlib.util.scatter.plot_45line(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.hline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.vline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.set_xylim(ax, vmin, vmax)
    #plt.legend(loc='best')
