import numpy as np

from . import model

# ===================================================================================================
def wrapper(ilat, ilon, sm, le):
    '''
    Args:
        ilat (int)
        ilon (int)
        sm (np.ndarray)
        le (np.ndarray)
    Returns:
        (tuple): ((int, int), (scalar,), (scalar,), ...)
    '''
    # except: RuntimeError
    # ValueError: Each lower bound must be strictly less than each upper bound.
    try:
        error001, param001 = model.est_001(sm, le)
    except (RuntimeError, ValueError):
        error001, param001 = (np.nan, np.nan), (np.nan,)

    try:
        error010, param010 = model.est_010(sm, le)
    except (RuntimeError, ValueError):
        error010, param010 = (np.nan, np.nan), (np.nan, np.nan)

    try:
        error110, param110 = model.est_110(sm, le)
    except (RuntimeError, ValueError):
        error110, param110 = (np.nan, np.nan), (np.nan, np.nan, np.nan)

    try:
        error011, param011 = model.est_011(sm, le)
    except (RuntimeError, ValueError):
        error011, param011 = (np.nan, np.nan), (np.nan, np.nan, np.nan)

    try:
        error111, param111 = model.est_111(sm, le)
    except (RuntimeError, ValueError):
        error111, param111 = (np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan, np.nan)

    return (
        (ilat, ilon),
        error001, param001,
        error010, param010,
        error110, param110,
        error011, param011,
        error111, param111
    )

def decode_futures(fs, map_shape):
    '''
    Args:
        fs ([Future])
        map_shape (tuple): (lat, lon)
    Returns:
        ({tuple: np.ma.MaskedArray}): {(str, str, int): np.ma.MaskedArray}
    '''
    nz = sum(model.DICT_MODEL_PARAMNUM.values()) + 2 * len(model.DICT_MODEL_PARAMNUM)
    arr = np.full((nz, map_shape[0], map_shape[1]), np.nan)

    for f in fs:
        pos, *eps = f.result()
        vals = [val for ep in eps for val in ep]
        arr[:,pos[0],pos[1]] = vals

    keys = []
    for _model in model.MODELS:
        keys += [(_model, 'error', i) for i in range(2)]
        keys += [(_model, 'param', i) for i in range(model.DICT_MODEL_PARAMNUM[_model])]
    return {key: np.ma.masked_invalid(arr[i]) for i, key in enumerate(keys)}
