import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from concurrent import futures
from netCDF4 import Dataset
from itertools import islice
import warnings
warnings.simplefilter('ignore', UserWarning)

from ncar import CMIP as cmip
from util import get_main_file_path

from .config import *
from . import csm
from . import file
from . import catalog
from . import in_subst
from . import forcing
from .file import BS_DIR
from . import enstat
from . import stat
from . import figure
from . import nc
from . import vs_atm
from . import misc
from . import prod

SM_VARIABLE = 'mrsos'
LE_VARIABLE = 'hfls'

# ===================================================================================================
def est_models(sm_arr, le_arr):
    '''
    Args:
        sm_arr (np.ma.MaskedArray [time,lat,lon])
        le_arr (np.ma.MaskedArray [time,lat,lon])
    Returns:
        dict_res ({any: np.ma.MaskedArray})
    '''
    is_land = ~sm_arr[0].mask

    x, y = np.where(is_land)
    fs = []
    with futures.ProcessPoolExecutor(max_workers=16) as executor:
    #if True:
        for ilat, ilon in zip(x, y):
        #for ilat, ilon in zip([50], [10]):
            fs += [executor.submit(
                csm.mputil.wrapper, ilat, ilon, sm_arr[:,ilat,ilon], le_arr[:,ilat,ilon]
            )]
            #ret = csm.mputil.wrapper(ilat, ilon, sm_arr[,ilat,ilon], le_arr[,ilat,ilon])
    return csm.mputil.decode_futures(fs, is_land.shape)

# ===================================================================================================
def write_stat_nc(dst_path, src_paths, var_names, samples):
    '''
    Args:
        dst_path (str)
        src_paths ([str,])
        var_names ([str,]): netCDF4 variable name
        samples (str)
    Writes:
        netCDF4
    '''
    dict_var = {var_name: [] for var_name in var_names}
    for path in src_paths:
        src_ds = xr.open_dataset(path)
        for var_name in var_names:
            dict_var[var_name] += [src_ds[var_name].to_masked_array()]
    for var_name in var_names:
        dict_var[var_name] = np.ma.stack(dict_var[var_name])

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with Dataset(dst_path, mode='w', format='NETCDF4') as dst_ds:
        dst_ds.samples = samples

        for dim_name in ('lat', 'lon'):
            src = src_ds[dim_name]
            dst_ds.createDimension(dim_name, src.size)
            var = dst_ds.createVariable(dim_name, src.dtype, (dim_name,))
            var[:] = src[:]

        for var_name, arr in dict_var.items():
            var = np.ma.stack(dict_var[var_name])
            mask = var.mask.sum(axis=0) >= len(var) * max_missing_rate
            mean = var.mean(axis=0)
            std = var.std(axis=0)
            cv = std / mean

            for arr, stat_name in zip((mean, std, cv), ('mean', 'std', 'cv')):
                var_name = f'{var_name}_{stat_name}'
                arr.mask = np.logical_or(arr.mask, mask)
                ncvar = dst_ds.createVariable(
                    var_name, arr.dtype, ('lat','lon'),
                    fill_value=csm.netcdf.FILL_VALUE,
                )
                ncvar[:] = arr[:].filled(csm.netcdf.FILL_VALUE)
                ncvar.standard_name = var_name
                ncvar.units = ''
    print(' ', os.path.basename(dst_path))

# ===================================================================================================
def catalog2dict_props(catalog, key_attrs=[], log=True):
    '''
    Args:
        catalog (pandas.DataFrame)
        key_attrs ([str,]): to identify the data
        log (bool)
    Returns:
        dict_attr_props ({(str,): {str: np.ma.MaskedArray}}):
            {(attributes): {property name: property map}}
    '''
    dict_attr_props = {}
    for _, sr in catalog.iterrows():
        key = tuple(sr[key_attr] for key_attr in key_attrs)
        path = sr['path']
        dict_attr_props[key] = {}
        ds = xr.open_dataset(path)
        if log: print(path)
        best_model = csm.model.select_best(ds)
        dict_attr_props[key]['model'] = best_model
        for prop, dict_ncvar in csm.prop.DICT_NCVAR_NAME.items():
            dict_attr_props[key][prop] = csm.model.select_best_param(ds, best_model, dict_ncvar)
    return dict_attr_props

# ===================================================================================================
def read_conv_cmip(catalog, var_name, time_range, log=True, **kwargs):
    '''
    Args:
        catalog (pandas.DataFrame)
        var_name (str)
        time_range (str)
        log (bool)
    Returns:
        da (xr.DataArray)
    '''
    table_id, variable_id = config.get_table_variable_ids(var_name)
    _catalog = cmip.catalog.filter(catalog, log=False,
        table_id=table_id, variable_id=variable_id, **kwargs)
    sr = _catalog.iloc[0]
    path = cmip.catalog.attr2path(sr, time_range=str(time_range), bs_dir=cmip.config.LOCAL_BS_DIR)
    da = xr.open_dataset(path)[variable_id]
    if log: print(path, da.shape)
    return da

# ===================================================================================================
# iter
# ===================================================================================================
def chunk_iter(iterable, size):
    '''
    Args:
        iterable (iterable)
        size (int)
    Yields:
        chunk (list)
    Examples:
        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        for chunk in chunked_iterable(lst, 3):
            print(chunk) # [1, 2, 3] -> [4, 5, 6] -> [7, 8]
    '''
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk

# ===================================================================================================
def read_ncs_vars(paths, var_names, log=False):
    '''
    Args:
        paths ([str,]): ['*.nc', ...]
        var_names ([str,])
        log (bool)
    Returns:
        dict_arr ({str: np.ma.MaskedArray}): {var_name: arr [len(paths),*]}
    '''
    dict_arr = {var_name: [] for var_name in var_names}
    for path in paths:
        ds = xr.open_dataset(path)
        for var_name in var_names:
            _arr = ds[var_name].to_masked_array()
            dict_arr[var_name] += [_arr]
        if log: print(path, _arr.shape)
    for k in dict_arr.keys():
        dict_arr[k] = np.ma.stack(dict_arr[k], axis=0)
    return dict_arr

# ===================================================================================================
def get_tmp_file_path(*file_names):
    main_path = get_main_file_path()
    dir_names = os.path.dirname(main_path).split('lsmconst/tool/')[-1].split('/')
    dir_names = [dir_name for dir_name in dir_names if not dir_name == '.']

    if file_names == []:
        file_name = 'tmp.pkl'
    else:
        dir_names += file_names[:-1]
        file_name = file_names[-1]
        root, ext = os.path.splitext(file_name)
        if ext == '': file_name = root + '.pkl'
    return os.path.join(BS_DIR, 'out', 'tmp', *dir_names, file_name)

# ===================================================================================================
def path_overlap_mask(source_id):
    return os.path.join(BS_DIR, 'out', 'mask', 'overlap', source_id + '.nc')


def read_overlap_mask(source_id):
    path = path_overlap_mask(source_id)
    return xr.open_dataset(path)