#! /usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset

import pdutil
from laiapy import csm
from ncar import CMIP as cmip

import lsmconst as ll

dir_name_src = 'csm_monthly'
dir_name_dst = 'prop_monthly'
cond = {
    'experiment_id': [
        'historical',    
        'land-hist', 'land-hist-cruNcep', 'land-hist-wfdei', 'land-hist-princeton',
    ],
    'grid_label': '2x2',
    'time_range': '190301-201212',
}
#OVER_WRITE = False
OVER_WRITE = True

tas_mask = True

# ===================================================================================================
def prop_write(src_path, dst_path, tas_path=None):
    '''
    Args:
        src_path (str)
        dst_path (str)
        tas_path (str)
    Writes:
        dst_ds (netCDF)
    '''
    src_ds = xr.open_dataset(src_path)
    print(src_path)
    dict_prop = {'model': csm.model.select_best(src_ds)}
    for prop_name, dict_ncvar in csm.prop.DICT_NCVAR_NAME.items():
        dict_prop[prop_name] = csm.model.select_best_param(src_ds, dict_prop['model'], dict_ncvar)

    if tas_path:
        tas_ds = xr.open_dataset(tas_path)
        mask = tas_ds['tas'].to_masked_array() < ll.config.TAIR_MIN
        for k in dict_prop.keys():
            dict_prop[k] = np.ma.masked_array(dict_prop[k], mask=mask)

    glob_attr = {
        'title': 'Properties of SM-LE',
    }
    var_attr = {}
    for prop_name in dict_prop.keys():
        var_attr[prop_name] = {
            'long_name': csm.prop.DICT_NETCDF_INFO[prop_name],
            'units': '',
        }
    var_attr['model']['fill_value'] = -9
    lat_name, lon_name = src_ds['BIC_001'].dims[-2:]
    ll.nc.write(dst_path, dict_prop, src_ds, glob_attr, var_attr, (lat_name, lon_name))


def get_in_subst(cmip_catalog, subst_catalog, sr):
    if sr['experiment_id'] in ll.config.CPL_EXPERIMENT_IDS:
        _catalog = pdutil.filter(subst_catalog, log=False,
            experiment_id=sr['experiment_id'], source_id=sr['source_id'])
        assert(len(_catalog) == 1)
        return _catalog.iloc[0]['path']
    elif sr['experiment_id'] in ll.config.OFL_EXPERIMENT_IDS:
        _catalog = pdutil.filter(cmip_catalog, log=False,
            experiment_id=sr['experiment_id'], source_id='forcing')
        assert(len(_catalog) == 1)
        return _catalog.iloc[0]['path']

# ===================================================================================================
def main(*args):
    catalog = ll.catalog.read(dir_name_src)
    catalog = pdutil.filter(catalog, **cond)

    if tas_mask:
        tas_catalog = cmip.catalog.read('local')
        tas_catalog = pdutil.filter(tas_catalog, table_id='clim', variable_id='tas', **cond)
        subst_tas_catalog = ll.catalog.read(ll.in_subst.DIR_NAME)

    dst_dir = ll.file.get_dir_path(dir_name_dst)
    args = []
    for _, sr in catalog.iterrows():
        src_path = sr['path']
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        if (not OVER_WRITE) and os.path.exists(dst_path): continue

        if tas_mask:
            _tas_catalog = pdutil.filter(tas_catalog, log=False,
                **{attr: sr[attr] for attr in ('source_id', 'experiment_id', 'member_id', 'grid_label', 'time_range')})

            if len(_tas_catalog) > 0:
                _tas_catalog = cmip.catalog.filter_latest_version(_tas_catalog)
                assert(len(_tas_catalog) == 1)
                tas_path = _tas_catalog.iloc[0]['path']
            else:
                print('NO TAIR -> subst', sr['experiment_id'], sr['source_id'], sr['member_id'])
                tas_path = get_in_subst(tas_catalog, subst_tas_catalog, sr)
                print(tas_path)

            args += [(src_path, dst_path, tas_path)]
        else:
            args += [(src_path, dst_path)]
    print(len(args))
    '''
    args = []
    src_dir = ll.file.get_dir_path(dir_name_src)
    dst_dir = ll.file.get_dir_path(dir_name_dst)
    for file_name in os.listdir(src_dir):
        #dict_attr = ll.file.parse_file_name(file_name)
        #if not dict_attr['member_id'] in member_ids: continue
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        if (not OVER_WRITE) and os.path.exists(dst_path): continue
        args += [(src_path, dst_path)]
    '''
    for _args in args:
        prop_write(*_args)
    return

if __name__=='__main__':
    main(*sys.argv)
    ll.catalog.write(dir_name_dst)