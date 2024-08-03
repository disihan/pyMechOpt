import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from matplotlib import rcParams

# from vald_init import *

from pyMechOpt.multi_obj import mo_problem
from pyMechOpt.basic import x2yaml
import shutil

so_dir = "../ls and soo/"


def copy_so(name):
    res_dir = so_dir + "./res_" + name + "/"
    # hist_dir = so_dir + "./hist_" + name + "/"
    res_file = res_dir + "mech.opt.yaml"
    shutil.copy2(res_file, "../mech/" + "mech.opt." + name + ".yaml")
    return
