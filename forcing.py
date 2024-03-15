import os
import sys
import numpy as np

from dictutil import PartialKeyDict

BS_DIR = '/glade/work/dtokuda/data/forcing'
DICT_ATTR = {
    'activity_id': 'LS3MIP',
    'institution_id': 'forcing',
    'source_id': 'forcing',
    'member_id': 'forcing',
    'table_id': 'clim',
    'grid_label': '2x2',
    'version': 'v99999999',
}
DICT_EXP = {
    'GSWP3': 'land-hist',
    'CRUNCEP': 'land-hist-cruNcep',
    'PRINCETON': 'land-hist-princeton',
    'WFDEI': 'land-hist-wfdei',
}
DICT_VAR = PartialKeyDict({
    ('LWdown', 'dlwrf'): 'rlds',
    ('SWdown', 'dswrf'): 'rsds',
    'Rainf': 'prc',
    'Snowf': 'prsn',
    ('Tair', 'tmp', 'tas'): 'tas',
})


# ===================================================================================================
def path_npy(dir_name, var_name, iyear, tres_name):
    return os.path.join(BS_DIR, dir_name, var_name, f'{var_name}_{iyear}_{tres_name}.npy')

def read_npy(dir_name, var_name, iyear, tres_name, log=True):
    path = path_npy(dir_name, var_name, iyear, tres_name)
    arr = np.load(path)[:,0]
    if log: print(path, arr.shape)
    return arr

# ---------------------------------------------------------------------------------------------------
def path_clim_nc(dir_name, var_name, syear, eyear, tres_name):
    return os.path.join(BS_DIR, dir_name, var_name, f'{var_name}_clim-{syear}01-{eyear}12_{tres_name}.nc')
