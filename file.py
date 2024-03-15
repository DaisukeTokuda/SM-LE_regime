import os

from ncar import CMIP as cmip

from .config import BS_DIR, SIM_ATTRS

MEMBER_ID_SUFFIX = 'members'

# ===================================================================================================
def get_dir_path(dir_name):
    return os.path.join(BS_DIR, 'out', dir_name)

def get_sim_name(dict_attr, **kwargs):
    _dict_attr = {}
    for attr in SIM_ATTRS:
        if attr in kwargs:
            _dict_attr[attr] = kwargs[attr]
        else:
            _dict_attr[attr] = dict_attr[attr]
    return '_'.join(list(_dict_attr.values()))

def get_file_path(dir_name, dict_attr, suffix='.nc', **kwargs):
    file_name = get_sim_name(dict_attr, **kwargs) + suffix
    return os.path.join(get_dir_path(dir_name), file_name)


def parse_file_name(file_name, suffix='.nc'):
    '''
    Args:
        file_name (str)
    Returns:
        ({str: str})
    '''
    return {k: v
        for k, v in zip(
            SIM_ATTRS, file_name.replace(suffix, '').split('_')
        )
    }

# ===================================================================================================
def int2member(num):
    return str(num) + MEMBER_ID_SUFFIX


def member2int(member_id):
    return int(member_id.split(MEMBER_ID_SUFFIX)[0])