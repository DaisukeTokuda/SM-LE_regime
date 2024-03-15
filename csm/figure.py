import matplotlib.pyplot as plt

from figlib import imshow_discrete_color

from .model import MODELS

DICT_MODEL_COLOR = {
    '001': '#54C87D',
    '010': '#CFBDE2',
    '110': '#FFC07B',
    '011': '#EDEBB1',
    '111': '#006EB9',
}
__XMIN, __XMAX = 0, 1

SM_UNIT = 'm$^3$/m$^3$'
LE_UNIT = 'W/m$^2$'

# ===================================================================================================
def scatter_sm_le(ax, sm, le):
    '''
    Args:
        ax (plt.Axes)
        sm (np.ndarray)
        le (np.ndarray)
    '''
    ax.scatter(sm, le, color='grey', alpha=0.3)

# ===================================================================================================
def plot(ax, model, param):
    '''
    Args:
        ax (plt.Axes)
        model (str)
        param (tuple)
    '''
    {
        '001': plot_001,
        '010': plot_010,
        '110': plot_110,
        '011': plot_011,
        '111': plot_111,
    }[model](ax, param)


def plot_001(ax, param):
    '''
    Args:
        ax (plt.Axes)
        param (tuple): (yconst,)
            yconst (scalar): constant y value
    '''
    model_name = '001'
    yconst = param[0]

    ax.hlines(yconst, __XMIN, __XMAX, color=DICT_MODEL_COLOR[model_name], label=model_name)


def plot_010(ax, param):
    '''
    Args:
        ax (plt.Axes)
        params (tuple): (intc, slp)
            intc (scalar): intercept
            slp (scalar): slope
    '''
    model_name = '010'
    intc, slp = param

    y0 = intc
    y1 = slp * __XMAX + intc
    ax.plot([__XMIN, __XMAX], [y0, y1], color=DICT_MODEL_COLOR[model_name], label=model_name)


def plot_110(ax, param):
    '''
    Args:
        ax (plt.Axes)
        param (scalar): (bpx, bpy, slp_right)
            bpx, bpy (scalar): Break point (x, y)
            slp_right (scalar): slope of the right segment
    '''
    model_name = '110'
    bpx, bpy, slp_right = param

    ax.hlines(bpy, __XMIN, bpx, color=DICT_MODEL_COLOR[model_name])
    y1 = slp_right * (__XMAX - bpx) + bpy
    ax.plot([bpx, __XMAX], [bpy, y1], color=DICT_MODEL_COLOR[model_name], label=model_name)


def plot_011(ax, param):
    '''
    Args:
        ax (plt.Axes)
        param (tuple): (bpx, bpy, slp_left)
            bpx, bpy (scalar): Break point (x, y)
            slp_left (scalar): slope of the left segment
    '''
    model_name = '011'
    bpx, bpy, slp_left = param

    y0 = slp_left * (__XMIN - bpx) + bpy
    ax.plot([__XMIN, bpx], [y0, bpy], color=DICT_MODEL_COLOR[model_name])
    ax.hlines(bpy, bpx, __XMAX, color=DICT_MODEL_COLOR[model_name], label=model_name)


def plot_111(ax, param):
    '''
    Args:
        ax (plt.Axes)
        params (tuple): (bp0x, bp0y, slp_mid, bp1x, bp1y)
            bp0x, bp0y (scalar): Break point 0 (x, y)
            slp_mid (scalar): slope of the middle segment
            bp1x, bp1y (scalar): Break point 1 (x, y)
    '''
    model_name = '111'
    bp0x, bp0y, _, bp1x, bp1y = param


    ax.hlines(bp0y, __XMIN, bp0x, color=DICT_MODEL_COLOR[model_name])
    ax.plot([bp0x, bp1x], [bp0y, bp1y], color=DICT_MODEL_COLOR[model_name], label=model_name)
    ax.hlines(bp1y, bp1x, __XMAX, color=DICT_MODEL_COLOR[model_name])

# ===================================================================================================
def imshow_best_model(ax, arr, *args, **kwargs):
    '''
    Args:
        ax (plt.Axes)
        arr (np.ndarray [lat,lon])
    '''
    color = ['#00000000']
    for model in MODELS: color += [DICT_MODEL_COLOR[model]]
    return imshow_discrete_color(ax, arr, [-9, 0, 1, 2, 3, 4], color, *args, **kwargs)