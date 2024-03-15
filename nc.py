import os
import numpy as np
from netCDF4 import Dataset

from laiapy import csm

from .config import BS_DIR

# ===================================================================================================
def write(dst_path, dict_arr, src_ds, glob_attr={}, var_attr={}, dim_names=('lat', 'lon')):
    '''
    Args:
        dst_path (str)
        dict_arr ({str: np.ma.MaskedArray})
        src_ds (xarray.DataSet)
        glob_attr ({})
        var_attr ({str: {str: any}}): attributes of each variable
        dim_names ((str,))
    Writes:
        dst_ds (NETCDF4)
    '''
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with Dataset(dst_path, mode='w', format='NETCDF4') as dst_ds:
        # some file-level meta-data attributes:
        for k, v in glob_attr.items(): setattr(dst_ds, k, v)

        for dim_name in dim_names:
            src = src_ds[dim_name]
            dst_ds.createDimension(dim_name, src.size)
            var = dst_ds.createVariable(dim_name, src.dtype, (dim_name))
            var[:] = src[:]

        for k, v in dict_arr.items():
            filval = csm.netcdf.get_fill_value(v.dtype)
            ncvar = dst_ds.createVariable(k, v.dtype, dim_names, fill_value=filval)
            if isinstance(v, np.ma.MaskedArray):
                ncvar[:] = v.filled(filval)
            else:
                ncvar[:] = v
            ncvar.standard_name = k
            if k in var_attr.keys():
                for _k, _v in var_attr[k].items(): setattr(ncvar, _k, _v)
    #print(dst_path.replace(BS_DIR, '').lstrip('/'))
    print(dst_path)


def write_without_source(dst_path, dict_arr, glob_attr={}, var_attr={}, dim_names=('lat', 'lon')):
    '''
    Args:
        dst_path (str)
        dict_arr ({str: np.ma.MaskedArray})
        glob_attr ({})
        var_attr ({str: {str: any}}): attributes of each variable
        dim_names ((str,))
    Writes:
        dst_ds (NETCDF4)
    '''
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with Dataset(dst_path, mode='w', format='NETCDF4') as dst_ds:
        for k, v in glob_attr.items(): setattr(dst_ds, k, v)

        for dim_name in dim_names:
            arr = dict_arr[dim_name]
            dst_ds.createDimension(dim_name, arr.size)
            var = dst_ds.createVariable(dim_name, arr.dtype, (dim_name))
            var[:] = arr[:]

        for k, v in dict_arr.items():
            if k in dim_names: continue
            ncvar = dst_ds.createVariable(k, v.dtype, dim_names)
            ncvar[:] = v
            ncvar.standard_name = k
            if k in var_attr.keys():
                for _k, _v in var_attr[k].items(): setattr(ncvar, _k, _v)
    print(dst_path)

# ===================================================================================================
def write_csm(dst_path, dict_arr, src_ds, dim_names=('lat', 'lon')):
    '''
    Args:
        dst_path (str)
        dict_arr ({any: np.ma.MaskedArray})
        src_ds (xarray.DataSet)
    Writes:
        (NETCDF4)
    '''
    glob_attr = {
        'title': 'Breakpoint of SM-LE',
        'history': '',
        'references': 'Hsu & Dirmeyer, 2022; 2023',
        'comment': '',
    }
    _dict_arr = {}
    var_attr = {}
    for k, v in dict_arr.items():
        std_name, long_name = csm.netcdf.DICT_NETCDF_INFO[k]
        _dict_arr[std_name] = v
        var_attr[k] = {
            'units': '',
            'long_name': long_name,
        }
    write(dst_path, _dict_arr, src_ds, glob_attr, var_attr, dim_names=dim_names)