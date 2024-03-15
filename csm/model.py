'''
Sources:
    https://github.com/hhsu81819/Soil-moisture-regime-and-projection/tree/2.0v
'''
import os
import sys
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf
from scipy import optimize

MODELS = (
    '001', '010', '110', '011', '111'
)
DICT_MODEL_PARAMNUM = {
    '001': 1,
    '010': 2,
    '110': 3,
    '011': 3,
    '111': 5,
}

__DICT_MODEL_BIC_PENALTY = {
    '001': 0,
    '010': 10,
    '110': 20,
    '011': 20,
    '111': 30,
}

# ===================================================================================================
def calc_degree_of_freedom(arr):
    '''
    Args:
        arr (np.ndarray)
    Returns:
        (scalar): Degree of Freedom
    '''
    lacc = acf(arr)[1] # (scalar) autocorrelation for 1-lag
    tau = -1.0 / np.log(np.sqrt(np.sum(lacc * np.abs(lacc)) / 40.0))
    return np.rint(float(len(arr)) / (tau + 1.0))


def est_001(sm, le):
    '''
    Args:
        sm (np.ndarray [time]): Soil moisture
        le (np.ndarray [time]): Latent heat flux
    Returns:
        error (tuple): (rss, bic)
            rss (scalar): residual sum of squares
            bic (scalar): Bayesian information criterion
        param (tuple): (yconst,)
            yconst (scalar): constant y value
    '''
    yconst = le.mean()
    rss = np.var(le) * len(sm)
    bic = len(sm) * np.log(rss / len(sm)) + np.log(len(sm))
    return (rss, bic), (yconst,)


def est_010(sm, le):
    '''
    Args:
        sm (np.ndarray [time]): Soil moisture
        le (np.ndarray [time]): Latent heat flux
    Returns:
        error (tuple): (rss, bic)
            rss (scalar): residual sum of squares
            bic (scalar): Bayesian information criterion
        param (tuple): (intc, slp)
            intc (scalar): intercept
            slp (scalar): slope
    '''
    x, y = sm.reshape(-1, 1), le.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    intc = model.intercept_[0]
    slp = model.coef_[0][0]

    y_pred = sm * slp + intc
    RSS_01 = np.sum(np.square(le - y_pred))
    BIC_01 = len(sm) * np.log(RSS_01 / len(sm)) + 2.0 * np.log(len(sm))
    return (RSS_01, BIC_01), (intc, slp)


def __piecewise_linear(x, x0, y0, k1, k2):
    """
    optimize.curve_fit does the heavy lifting. p & e are 4-element series of results and error/uncertainties for:
    [0] : X or soil moisture value for breakpoint
    [1] : Y (T, flux, etc.) value for breakpoint
    [2] : Slope on left side of breakpoint
    [3] : slope on right side of breakpoint
    __piecewise_linear is defined above - the function to optimize over
    Next 2 arguments are X and Y series (sorted on X) of daily data
    p0 is an optional first guess for each of the 4 predicted parameters
    bounds sets limits on the acceptable ranges of each parameter
    """
#   Compute optimization of fitting two connected linregs to data.
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def __piecewise3sg_linear(x, x0, x1, y0, y1, k1, k2, k3):
    """
    Hypothesis: Fitting 3-segment regression model to the data assume there are a wilting point and a critical point
    lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0]
    lambda x: k1*x + b, lambda x: k1*x + b + k2*(x-x0), lambda x: k1*x + b + k2*(x-x0) + k3*(x - x1)
    Note that this could generate results that x1<x0. This is not solved analytically but by removing the result of 
    this afterward.
    """
    condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    funclist = [lambda x:k1*(x-x0) + y0, lambda x:k2*(x-x0) + y0, lambda x:k2*(x1-x0) + y0 + k3*(x-x1)]
    return np.piecewise(x, condlist, funclist)


def est_110(sm, le):
    '''
    Args:
        sm (np.ndarray [time]): Soil moisture
        le (np.ndarray [time]): Latent heat flux
    Returns:
        error (tuple): (rss, bic)
            rss (scalar): residual sum of squares
            bic (scalar): Bayesian information criterion
        param (scalar): (bpx, bpy, slp_right)
            bpx, bpy (scalar): Break point (x, y)
            slp_right (scalar): slope of the right segment
    '''
    p, _ = optimize.curve_fit(
        __piecewise_linear, sm, le,
        p0=[(np.max(sm) + np.min(sm)) * 0.5, np.median(le), 0, 50],
        bounds=([np.min(sm), np.min(le), -0.001, 1], [np.max(sm), np.max(le), 0.001, 1000.0])
        )
    bpx, bpy, _, slp_right = p
    y_pred = np.empty(sm.shape)
    mask = sm < bpx
    y_pred[mask] = bpy
    mask = bpx <= sm
    y_pred[mask] = (sm[mask] - bpx) * slp_right + bpy
    rss = np.sum(np.square(le - y_pred))
    bic = len(sm) * np.log(rss / len(sm)) + 4.0 * np.log(len(sm))
    return (rss, bic), (bpx, bpy, slp_right)


def est_011(sm, le):
    '''
    Args:
        sm (np.ndarray [time]): Soil moisture
        le (np.ndarray [time]): Latent heat flux
    Returns:
        error (tuple): (rss, bic)
            rss (scalar): residual sum of squares
            bic (scalar): Bayesian information criterion
        param (tuple): (bpx, bpy, slp_left)
            bpx, bpy (scalar): Break point (x, y)
            slp_left (scalar): slope of the left segment
    '''
    p, _ = optimize.curve_fit(
        __piecewise_linear, sm, le,
        p0=[(np.max(sm) + np.min(sm)) * 0.5, np.median(le), 1, 0],
        bounds=([np.min(sm), np.min(le), 0, -0.001], [np.max(sm), np.max(le), 1000, 0.001])
        )
    bpx, bpy, slp_left, _ = p
    y_pred = np.empty(sm.shape)
    mask = sm < bpx
    y_pred[mask] = (sm[mask] - bpx) * slp_left + bpy
    mask = bpx <= sm
    y_pred[mask] = bpy

    rss = np.sum(np.square(le - y_pred))
    bic = len(sm) * np.log(rss / len(sm)) + 4.0 * np.log(len(sm))
    return (rss, bic), (bpx, bpy, slp_left)


def est_111(sm, le):
    '''
    Args:
        sm (np.ndarray [time]): Soil moisture
        le (np.ndarray [time]): Latent heat flux
    Returns:
        error (tuple): (rss, bic)
            rss (scalar): residual sum of squares
            bic (scalar): Bayesian information criterion
        param (tuple): (bp0x, bp0y, slp_mid, bp1x, bp1y)
            bp0x, bp0y (scalar): Break point 0 (x, y)
            slp_mid (scalar): slope of the middle segment
            bp1x, bp1y (scalar): Break point 1 (x, y)
    '''
    p, _ = optimize.curve_fit(
        __piecewise3sg_linear, sm, le,
        p0=[(np.max(sm + 0.001) + np.min(sm)) * 0.5, (np.max(sm + 0.001) + np.min(sm)) * 0.5,
            np.median(le), np.median(le), 0, 50, 0],
        bounds=([np.min(sm), np.min(sm), np.min(le), np.min(le), -0.001, 1, -0.001],
            [np.max(sm + 0.001), np.max(sm + 0.001), np.max(le) + 0.001, np.max(le) + 0.001, 0.001, 1000.0, 0.001])
        )
    bp0x, bp1x, bp0y, _, _, slp_mid, _ = p
    bp1y = bp0y + slp_mid * (bp1x - bp0x)

    y_pred = np.empty(sm.shape)
    mask = sm < bp0x
    #y_pred[mask] = sm[mask] - bp0x + bp0y
    y_pred[mask] = bp0y
    mask = np.logical_and(bp0x <= sm, sm < bp1x)
    y_pred[mask] = bp1y + slp_mid * (sm[mask] - bp1x)
    mask = bp1x <= sm
    #y_pred[mask] = bp1y + sm[mask] - bp1x
    y_pred[mask] = bp1y

    rss = np.sum(np.square(le - y_pred))
    bic = len(sm) * np.log(rss / len(sm)) + 6.0 * np.log(len(sm))
    return (rss, bic), (bp0x, bp0y, slp_mid, bp1x, bp1y)

# ===================================================================================================
def __ds2arr(ds, variable):
    '''
    Args:
        ds (xarray.core.dataset.Dataset or netcdf4.Dataset)
    Returns:
        arr (np.ma.MaskedArray)
    '''
    try:
        return ds[variable].to_masked_array()
    except:
        return ds.variables[variable][:]


def select_best(ds):
    '''
    Args:
        ds (netcdf4.Dataset or xarray.core.dataset.Dataset)
    Returns:
        minbic_model_idx (np.ndarray [lat,lon]): index compatible with MODELS
            no value: -9 (e.g., ocean)
    '''
    bic = []
    for model in MODELS:
        _bic = __ds2arr(ds, f'BIC_{model}')
        _bic += __DICT_MODEL_BIC_PENALTY[model]
        if model == '010':
            _slp = __ds2arr(ds, 'Slope_1Seg')
            _bic.mask = np.logical_or(_bic.mask, _slp < 0)
        elif model == '111':
            _bp0x = __ds2arr(ds, 'BPx1_3Seg')
            _bp1x = __ds2arr(ds, 'BPx2_3Seg')
            _bic.mask = np.logical_or(_bic.mask, _bp0x > _bp1x)
        bic += [_bic]
    bic = np.ma.stack(bic)
    minbic = np.ma.min(bic, axis=0)

    minbic_model_idx = np.full(minbic.shape, -9).astype('int8')
    mask = []
    for i, _bic in enumerate(bic):
        mask += [np.ma.getmaskarray(_bic)]
        minbic_model_idx[np.ma.where(_bic == minbic)] = i
    return minbic_model_idx


def select_best_param(ds, best_model_index, dict_model_param_name):
    '''
    Args:
        ds (xarray.core.dataset.Dataset)
        best_model_index (np.ndarray [lat,lon])
        dict_model_param_name ({str: str}): {model name: variable name of the parameter}
    Returns:
        (np.ma.MaskedArray [lat,lon])
    '''
    arr = np.full(best_model_index.shape, np.nan)
    for i, model in enumerate(MODELS):
        if not model in dict_model_param_name: continue
        is_best = best_model_index == i
        arr[is_best] = ds[dict_model_param_name[model]].to_masked_array()[is_best]
    return np.ma.masked_invalid(arr)
