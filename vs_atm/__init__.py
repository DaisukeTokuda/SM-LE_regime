import os
import pandas as pd
import numpy as np
import xarray as xr

import pdutil

from . import bpdef
from . import sens
from . import fig

from .. import config as llconf
from .. import file as llfile

MODEL_DIR_NAME = 'vs_atm/model'
SCORE_COLUMNS = ['model_' + v for v in llconf.SIM_ATTRS]
SCORE_COLUMNS += ['prop_name', 'class_id']
SCORE_COLUMNS += ['target_' + v for v in llconf.SIM_ATTRS]
SCORE_COLUMNS += ['stat_name', 'value']

IMPORTANCE_COLUMNS = llconf.SIM_ATTRS + ['prop_name', 'class_id', 'variable_id', 'stat_name', 'type', 'value']

def get_model_name(model_type, ind_variable_ids):
    return model_type + '_' + '-'.join(ind_variable_ids)

def get_dir_name(model_type, ind_variable_ids):
    model_name = get_model_name(model_type, ind_variable_ids)
    return 'atm_' + model_name

def get_model_path(dir_name, dict_attr, **kwargs):
    return llfile.get_file_path(dir_name, dict_attr, suffix='.pkl', **kwargs)

def get_score_path(model_dir_name):
    return os.path.join(llconf.BS_DIR, 'out', 'atm_score', model_dir_name + '.csv') 

def get_importance_path(model_name):
    return os.path.join(llconf.BS_DIR, 'out', 'atm_importance', model_name + '.csv') 

def read_importance(model_type, ind_variable_ids, log=True):
    model_name = get_dir_name(model_type, ind_variable_ids)
    src_path = get_importance_path(model_name)
    df = pd.read_csv(src_path)
    if log: print(src_path, df.shape)
    return df

# ===================================================================================================
def get_predmap_dir_name(model_type, ind_variable_ids):
    model_name = get_model_name(model_type, ind_variable_ids)
    return f'atm_{model_name}_predmap'

def get_predmap_path(dir_name, model_attr, target_attr, suffix='.nc'):
    file_name = llfile.get_sim_name(model_attr) + '_' + llfile.get_sim_name(target_attr) + suffix
    return os.path.join(llfile.get_dir_path(dir_name), file_name)


# ===================================================================================================
def find_sr_with_subst(cmip_catalog, subst_catalog, sim, ind_variable_ids):
    if isinstance(ind_variable_ids, str):
        ind_variable_ids = [ind_variable_ids]
        return_sr = True
    else:
        return_sr = False

    ind_srs = []
    for variable_id in ind_variable_ids:
        _cmip_catalog = pdutil.filter(cmip_catalog, log=False,
            variable_id=variable_id, **sim.to_dict())
        if len(_cmip_catalog) > 0:
            ind_srs += [_cmip_catalog.iloc[0]]
            continue
        _subst_catalog = pdutil.filter(subst_catalog, log=False,
            variable_id=variable_id, **sim[llconf.FORCING_MATCH_ATTRS].to_dict())
        if len(_subst_catalog) > 0:
            assert(len(_subst_catalog) == 1)
            ind_srs += [_subst_catalog.iloc[0]]

    if len(ind_variable_ids) != len(ind_srs):
        raise ValueError()
    if return_sr:
        return ind_srs[0]
    else:
        return ind_srs


def find_sr(cmip_catalog, sim, variable_ids):
    if isinstance(variable_ids, str):
        variable_ids = [variable_ids]
        return_sr = True
    else:
        return_sr = False

    srs = []
    for variable_id in variable_ids:
        _cmip_catalog = pdutil.filter(cmip_catalog, log=False,
            variable_id=variable_id, **{k: sim[k] for k in llconf.SIM_ATTRS})
        if len(_cmip_catalog) > 0:
            srs += [_cmip_catalog.iloc[0]]

    if len(variable_ids) != len(srs):
        raise KeyError(sim)
    if return_sr: srs = srs[0]
    return srs

# ===================================================================================================
def encode_for_pred(smle_sr, ind_srs, ta_sr):
    '''
    Returns:
        (x, y) ((np.ndarray, np.ndarray) [n_grid])
        is_land (np.ndarray bool [lat, lon])
        X (np.ndarray (n_samples, n_independent_variables))
    '''
    ds = xr.open_dataset(smle_sr['path'])
    is_land = ~(ds['RSS_001'].to_masked_array().mask)

    ind_maps = [
        xr.open_dataset(ind_sr['path'])[ind_sr['variable_id']].to_masked_array()
        for ind_sr in ind_srs
    ]
    ta_map = xr.open_dataset(ta_sr['path'])[ta_sr['variable_id']].to_masked_array()
    is_warm = ta_map >= llconf.TAIR_MIN

    mask = np.logical_and(is_land, is_warm)
    x, y = np.where(mask)
    X = np.ma.stack([ind_map[x, y] for ind_map in ind_maps], axis=1)
    return (x, y), is_land, X