import os 
import sys
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

import figlib
from util import path as utpath
from util import stat as utstat
from dictutil import MultiGetDict

from . import misc
from .config import PROP_NAMES

PMIN, PMAX = 20, 80

DICT_PROP_NAME = {
    'wilt': 'Wilting Point',
    'slope': 'SM-LE Slope',
    'csm': 'Critical SM',
    'lesat': 'Saturated LE',
}
SM_UNIT = 'm$^3$/m$^3$'
LE_UNIT = 'W/m$^2$'
DICT_PROP_UNIT = {
    'wilt': SM_UNIT,
    'slope': f'{LE_UNIT} / {SM_UNIT}',
    'csm': SM_UNIT,
    'lesat': LE_UNIT,
}
DICT_PROP_NAMEUNIT = {
    prop_name: f'{DICT_PROP_NAME[prop_name]} ({DICT_PROP_UNIT[prop_name]})'
    for prop_name in PROP_NAMES
}

DICT_PROP_BPNAME = {
    'wilt': 'Dry - Transitional Breakpoint',
    'csm': 'Transitional - Wet Breakpoint',
}
DICT_VARIABLE_SHORTNAME = {
    'pr': 'P',
    'rds': 'R',
}

DICT_STAT_OPT = MultiGetDict({
    'num': {
        'vmin': 0,
    },
    'corr': {
        'cmap': 'RdBu',
        'vmin': -1,
        'vmax': 1,
    },
    'corr-p': {
        'vmin': 0,
        'vmax': 1,
    },
    'slope': {
        'cmap': 'RdBu',
        'vcenter': 0,
    },
    'rmsd': {
        'vmin': 0,
        'cmap': 'Reds',
    },
    'jaccard': {
        'vmin': 0,
        'vmax': 1,
        'cmap': 'Reds_r',
    }
})

# ===================================================================================================
def get_dir(*args):
    import __main__
    file_prefix = os.path.basename(__main__.__file__).replace('.py', '')
    return utpath.figure_dir(dir_names=['%date', file_prefix] + list(args), makedirs=True)

# ===================================================================================================
def get_stat_opt(stat_name):
    return DICT_STAT_OPT.get(stat_name, {})

# ===================================================================================================
# cartopy
# ===================================================================================================
def draw_worldmap(ax):
    land = cfeature.NaturalEarthFeature('physical', 'land', 
        facecolor='black', scale='110m',
        zorder=1)
    ax.add_feature(land)
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', 
        facecolor='lightgrey', scale='110m',
        zorder=1)
    ax.add_feature(ocean)
    ax.coastlines(resolution='110m', lw=0.5, zorder=99)
    return ax


def set_extent(ax):
    return ax.set_extent((0, 360.1, -65, 80), crs=ccrs.PlateCarree())


def colorbar(ax, im, loc='bottom', **kwargs):
    if loc == 'bottom':
        _kwargs = {'orientation': 'horizontal', 'extend': 'both'}
        pos = (0, -0.08, 1, 0.05)
    if loc == 'right':
        _kwargs = {'orientation': 'vertical', 'extend': 'both'}
        pos = (1.02, 0.1, 0.03, 0.8)
    ax_cb = figlib.axesutil.add_axes(ax, pos)
    _kwargs.update(kwargs)
    plt.colorbar(im,
        cax=ax_cb, **_kwargs)
    ax_cb.set_zorder(99)


def cp_imshow(ax, arr, **kwargs):
    if 'vmin' in kwargs and 'vcenter' in kwargs and 'vmax' in kwargs:
        vmin = kwargs.pop('vmin')
        vcenter = kwargs.pop('vcenter')
        vmax = kwargs.pop('vmax')
        vmax = max(abs(vmin), abs(vmax))
        norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=-vmax, vmax=vmax)
        kwargs.update({'norm': norm})
    ax = figlib.axesutil.replace_axes(ax, projection=ccrs.PlateCarree())
    draw_worldmap(ax)
    im = ax.imshow(
        arr,
        interpolation='nearest',
        origin='lower', extent=(-1, 359, -90, 90), transform=ccrs.PlateCarree(),
        zorder=9,
        **kwargs
    )
    set_extent(ax)
    return ax, im


def scatter_same_lim(ax, x, y, vmin, vmax, texts=None, *args, **kwargs):
    x, y = utstat.filter_invalid(x, y)
    ax.scatter(x, y, *args, **kwargs, zorder=9)

    if texts is not None: figlib.util.text_loc(ax, texts, loc='best')

    figlib.util.scatter.plot_45line(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.hline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.vline(ax, vmin=vmin, vmax=vmax, color='k', lw=1, zorder=1)
    figlib.util.scatter.set_xylim(ax, vmin, vmax)

# ===================================================================================================
# GridSpecFig
# ===================================================================================================
def cp_imshows(gsfig, start_row=1, start_col=1, share_cb=None):
    if share_cb == 'rows':
        for icol in range(start_col, gsfig.ncols):
            for irow in range(start_row, gsfig.nrows):
                ax = gsfig[irow, icol]
                arr = gsfig.data[irow, icol]
                if arr is None:
                    ax.remove()
                    continue
                _, im = cp_imshow(ax, arr, **gsfig.option[irow, icol].data)

                if gsfig.is_bottom(irow): colorbar(ax, im, loc='bottom')
                ax.axis('off')

    elif share_cb == 'cols':
        for irow in range(start_row, gsfig.nrows):
            for icol in range(start_col, gsfig.ncols):
                ax = gsfig[irow, icol]
                _, im = cp_imshow(ax, gsfig.data[irow, icol], **gsfig.option[irow, icol].data)

                if gsfig.is_right(icol): colorbar(ax, im, loc='right')
                ax.axis('off')

    else:
        for icol in range(start_col, gsfig.ncols):
            for irow in range(start_row, gsfig.nrows):
                ax = gsfig[irow, icol]
                _, im = cp_imshow(ax, gsfig.data[irow, icol], **gsfig.option[irow, icol].data)
                colorbar(ax, im, loc='bottom')
                ax.axis('off')


def scatters(gsfig, start_row=1, start_col=1, *args, **kwargs):
    for icol in range(start_col, gsfig.ncols):
        for irow in range(start_row, gsfig.nrows):
            ax = gsfig[irow, icol]
            x, y = gsfig.data[irow, icol]
            option = gsfig.option[irow, icol]
            vmin, vmax = option['vmin'], option['vmax']
            scatter_same_lim(ax, x, y, vmin, vmax, *args, **kwargs)

# ===================================================================================================
# Template
# ===================================================================================================
def cp_imshow_files(paths, var_names, texts_top, texts_left, text_00=None,
        opt_rows={}, opt_cols={}, share_norm=None):
    '''
    Args:
        paths ([str,])
        stat_names ([str])
    Returns:
        fig (plt.Figure)
        gs (plt.GridSpec)
    Draws:
        row: paths
        col: variables
    '''
    nrows = len(paths) + 1
    ncols = len(var_names) + 1

    width_ax, height_ax = 3.5, 1.5
    figsize = (width_ax * ncols, height_ax * nrows)
    gsfig = figlib.GridSpecFig(figsize, nrows, ncols, wspace=0.05)

    for irow, path in enumerate(paths, start=1):
        ds = xr.open_dataset(path)
        for icol, var_name in enumerate(var_names, start=1):
            gsfig.data[irow, icol] = ds[var_name].to_masked_array()

    if share_norm == 'rows':
        for icol, var_name in enumerate(var_names, start=1):
            vmin, vmax = misc.get_percentile_vminmax(gsfig.data[1:,icol], PMIN, PMAX)
            gsfig.option[:,icol].update({'vmin': vmin, 'vmax': vmax})
    elif share_norm == 'cols':
        for irow in range(1, nrows):
            vmin, vmax = misc.get_percentile_vminmax(gsfig.data[irow,1:], PMIN, PMAX)
            gsfig.option[irow,:].update({'vmin': vmin, 'vmax': vmax})
    if not opt_rows == {}:
        for irow, opt in opt_rows.items(): gsfig.option[irow, :].update(opt)
    if not opt_cols == {}:
        for icol, opt in opt_cols.items(): gsfig.option[:, icol].update(opt)

    gsfig.text_cols(0, texts_top)
    gsfig.text_rows(0, texts_left)
    if not text_00 is None:
        gsfig.text(0, 0, text_00)
    else:
        gsfig[0,0].remove()
    cp_imshows(gsfig, 1, 1, share_cb=share_norm)
    return gsfig


def scatter_files(ref_paths, tgt_paths, var_names, texts_top, texts_left, text_00=None,
        opt_rows={}, opt_cols={}, **kwargs):
    nrows = len(ref_paths) + 1
    ncols = len(var_names) + 1

    width_ax, height_ax = 3, 3
    figsize = (width_ax * ncols, height_ax * nrows)
    gsfig = figlib.GridSpecFig(figsize, nrows, ncols, wspace=0.25)

    pmin, pmax = 5, 95
    for irow, (ref_path, tgt_path) in enumerate(zip(ref_paths, tgt_paths), start=1):
        ref_ds = xr.open_dataset(ref_path)
        tgt_ds = xr.open_dataset(tgt_path)
        for icol, var_name in enumerate(var_names, start=1):
            x = ref_ds[var_name].to_masked_array().flatten()
            y = tgt_ds[var_name].to_masked_array().flatten()
            gsfig.data[irow, icol] = (x, y)

            vmin, vmax = misc.get_percentile_vminmax((x, y), pmin, pmax)
            gsfig.option[irow,icol].update({'vmin': vmin, 'vmax': vmax})

    if not opt_rows == {}:
        for irow, opt in opt_rows.items(): gsfig.option[irow, :].update(opt)
    if not opt_cols == {}:
        for icol, opt in opt_cols.items(): gsfig.option[:, icol].update(opt)

    gsfig.text_cols(0, texts_top)
    gsfig.text_rows(0, texts_left)
    if not text_00 is None:
        gsfig.text(0, 0, text_00)
    else:
        gsfig[0,0].remove()
    scatters(gsfig, 1, 1, **kwargs)
    return gsfig

# ===================================================================================================
def heatmap(ax, heatmap_df, cmap_name=None, **kwargs):
    '''
    Args:
        ax (plt.Axes)
        heatmap_df (pd.DataFrame)
    '''
    cmap = mpl.cm.get_cmap(cmap_name)
    cmap.set_bad('grey')
    ax = sns.heatmap(heatmap_df, cmap=cmap, mask=np.isnan(heatmap_df.to_numpy()), **kwargs)
    ax.xaxis.tick_top()
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

# ===================================================================================================
def bars_with_std(ax, arrs, labels=[], fix_zero=True, **kwargs):
    side_buffer = 0.1

    width = (1 - 2 * side_buffer) / len(arrs)

    x, height, yerr = [], [], []
    _x = side_buffer * 0.5 * width
    for arr in arrs:
        x += [_x]
        height += [arr.mean()]
        if len(arr) > 1:
            yerr += [np.std(arr)]
        else:
            yerr += [0]
        _x += width

    bar_kwargs = {
        'yerr': yerr,
        'color': 'royalblue', 'alpha': 0.5,
        'edgecolor': 'blue', 'width': width, 'align': 'center',
    }
    bar_kwargs.update(kwargs)
    ax.bar(x, height, **bar_kwargs)
    if not fix_zero:
        height = np.array(height)
        yerr = np.array(yerr)
        vmin = np.nanmin(height - yerr)
        vmax = np.nanmax(height + yerr)
        diff = vmax - vmin
        vmin -= 0.05 * diff
        vmax += 0.05 * diff
        #print(height, yerr)
        #print(min(height - yerr), max(height + yerr))
        if np.isnan(vmin) or np.isnan(vmax):
            pass
        else:
            ax.set_ylim(vmin, vmax)

    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

# ===================================================================================================
def bar_groups(ax, dict_group_arrs):
    '''
    Args:
        ax (plt.Axes)
        dict_group_arrs ({str: [np.ndarray]})
    '''
    interval = 0.3 # between groups
    buffer = 0.2 # from left and right edges

    n_arr = sum([len(v) for v in dict_group_arrs.values()])
    width = (1 - 2 * buffer - interval * (len(dict_group_arrs) - 1)) / n_arr

    x, height, yerr = [], [], []
    xticks, xticklabels = [], []
    _x = buffer + 0.5 * width
    for group_name, arrs in dict_group_arrs.items():
        group_xs = []
        for arr in arrs:
            x += [_x]
            group_xs += [_x]
            height += [arr.mean()]
            yerr += [np.std(arr)]
            _x += width
        _x += interval
        xticks += [np.array(group_xs).mean()]
        xticklabels += [group_name]

    ax.bar(x, height, yerr=yerr,
        color='royalblue', alpha=0.5, edgecolor='blue', width=width, align='center')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    xmin, xmax = 0, 1
    ax.hlines(0, xmin, xmax, color='k', lw=1)
    ax.set_xlim(xmin, xmax)