import os
import matplotlib.pyplot as plt

import ioutil
import util.path as utpath

PICKLE_DIR = '/glade/work/dtokuda/out/prod/lsm_const'
PICKLE_SUF = '.pkl'


DICT_STD_RANGE = {
    #'wilt_std': (0, 0.014),
    'wilt': (0, 0.10),
    #'csm_std': (0, 0.014),
    'csm': (0, 0.10),
    #'lesat_std': (0, 3.2),
    'lesat': (0, 35),
    #'slope_std': (0, 100),
    'slope': (0, 500),
}

DICT_MEAN_RANGE = {
    'wilt': (0, 0.4),
    'csm': (0, 0.4),
    #'slope_mean': (200, 500),
    'slope': (200, 700),
    'lesat': (25, 150),
}

DICT_STAT_LONGNAME = {
    'mean': 'Mean',
    'std': 'Standard Deviation',
    'num': 'Sample Fraction',
}
# ===================================================================================================
def get_main_file_prefix():
    import __main__
    return os.path.basename(__main__.__file__).replace('.py', '')

def get_pickle_path(*args):
    file_prefix = get_main_file_prefix()
    if not args == (): file_prefix += '-' + ('_').join(args)
    return os.path.join(PICKLE_DIR, file_prefix + PICKLE_SUF)

def dump_pickle(data, *args):
    path = get_pickle_path(*args)
    ioutil.dump_pickle(path, data, log=True)

def load_pickle(*args):
    path = get_pickle_path(*args)
    return ioutil.load_pickle(path, log=True)

# ===================================================================================================
def savefig(*args):
    file_prefix = get_main_file_prefix()
    if not args == (): file_prefix += '-' + ('_').join(args)
    fig_dir = utpath.figure_dir(dir_names=['prod', 'lsm_const'], makedirs=True)
    fig_path = os.path.join(fig_dir, file_prefix + '.png')
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05)
    print(fig_path)


def savefig_poster(*args):
    file_prefix = get_main_file_prefix()
    if not args == (): file_prefix += '-' + ('_').join(args)
    fig_dir = utpath.figure_dir(dir_names=['prod_poster', 'lsm_const'], makedirs=True)
    for suffix in ('.png', '.pdf'):
        fig_path = os.path.join(fig_dir, file_prefix + suffix)
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05)
        print(fig_path)
