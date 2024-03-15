import numpy as np

def get_percentile_vminmax(arrs, pmin, pmax):
    vmin, vmax = [], []
    for arr in arrs:
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(np.nan)
        _vmin = np.nanpercentile(arr, pmin)
        _vmax = np.nanpercentile(arr, pmax)
        if not np.isnan(_vmin): vmin += [_vmin]
        if not np.isnan(_vmax): vmax += [_vmax]
    return min(vmin), max(vmax)