BS_DIR = '/glade/work/dtokuda/project/lsmconst'

PROP_NAMES = ['wilt', 'csm', 'lesat', 'slope']
OFL_EXPERIMENT_IDS = [
    'land-hist',
    'land-hist-cruNcep',
    'land-hist-princeton',
    #'land-hist-wfdei',
]
CPL_EXPERIMENT_IDS = [
    'historical',
]
SIM_ATTRS = ['source_id', 'experiment_id', 'member_id', 'grid_label', 'time_range']
FORCING_MATCH_ATTRS = ['experiment_id', 'grid_label', 'time_range']

CPL_SOURCE_IDS = [
    'ACCESS-ESM1-5',
    'CESM2',
    'CMCC-CM2-SR5',
    'CNRM-CM6-1',
    'CanESM5',
    'EC-Earth3',
    'EC-Earth3-Veg',
    'GISS-E2-1-G',
    'GISS-E2-1-H',
    'INM-CM5-0',
    'IPSL-CM6A-LR',
    'MIROC-ES2L',
    'MIROC6',
    'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR',
    'MRI-ESM2-0',
    'UKESM1-0-LL',
]

OFL_SOURCE_IDS = [
    'CESM2',
    'CNRM-CM6-1',
    'CNRM-ESM2-1',
    'HadGEM3-GC31-LL',
    'IPSL-CM6A-LR',
    'MIROC6',
    'MPI-ESM1-2-LR',
    'UKESM1-0-LL',
]

CPLOFL_SOURCE_IDS = [
    'CESM2',
    'CNRM-CM6-1',
    'EC-Earth3-Veg',
    'IPSL-CM6A-LR',
    'MIROC6',
    'MPI-ESM1-2-LR',
    'UKESM1-0-LL',
]

DICT_TALBLE_ID = {
    'sm': 'Lmon',
    'le': 'Amon',
    'pr': 'Amon',
    'rn': 'Amon',
    'sd': 'Amon',
    'ld': 'Amon',
    'su': 'Amon',
    'lu': 'Amon',
}

DICT_VARIABLE_ID = {
    'sm': 'mrsos',
    'le': 'hfls',
    'pr': 'pr',
    'rn': 'rnet',
    'sd': 'rsds',
    'ld': 'rlds',
    'su': 'rsus',
    'lu': 'rlus',
}

SINGLE_MEMBER_IDS = ['r1i1p1f1', 'r1i1p1f2', 'r1i1p1f3', 'r2i1p1f1', 'r1i1p11f1', 'r10i1p1f1']

TAIR_MIN = 273.15 + 5 # [K]

OCEAN_FLAG = -2
COLD_FLAG = -1

def get_table_ids(*var_names):
    '''
    Args:
        var_names ([str]): ['sm', 'le', ...]
    Returns:
        ret ((str,)): (variable_id, ...)
    '''
    ret = []
    for var_name in var_names:
        ret += [DICT_TALBLE_ID[var_name]]
    if len(var_names) == 1:
        return ret[0]
    else:
        return tuple(ret)


def get_variable_ids(*var_names):
    '''
    Args:
        var_names ([str]): ['sm', 'le', ...]
    Returns:
        ret ((str,)): (variable_id, ...)
    '''
    ret = []
    for var_name in var_names:
        ret += [DICT_VARIABLE_ID[var_name]]
    if len(var_names) == 1:
        return ret[0]
    else:
        return tuple(ret)


def get_table_variable_ids(*var_names):
    '''
    Args:
        var_names ([str]): ['sm', 'le', ...]
    Returns:
        ret ((str, str)): ((table_id, variable_id), ...)
    '''
    ret = []
    for var_name in var_names:
        ret += [(DICT_TALBLE_ID[var_name], DICT_VARIABLE_ID[var_name])]
    if len(var_names) == 1:
        return tuple(ret[0])
    else:
        return tuple(ret)