import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from vald_init import *

from pyMechOpt.multi_obj import mo_problem


def vald_mo(name, name_cap):
    hist_f, time = mo_problem.load_hist(hist_dir="./hist_" + name + "/")
    res_f, orig_f = mo_problem.load_res(res_dir="./res_" + name + "/", hist_dir="./hist_" + name + "/", load_orig=True)

    fig_1, t_lns = mo_problem.plot_hist(hist_f)

    fig_21, fig_22, t_lns_21, t_lns_22 = mo_problem.plot_parallel_coordinates(res_f, orig_f)

    fig_3, t_lns, f_geom, f_mean, f_min, f_max = mo_problem.plot_hist_geom(hist_f)

    for fig in [fig_1,fig_21,fig_22,fig_3]:
        fig.set_size_inches(a, b)
        fig.tight_layout()

    fig_1.savefig(dir_save + name_cap + "-12-hist" + pic_type)
    fig_21.savefig(dir_save + name_cap + "-12-ratio" + pic_type)
    fig_22.savefig(dir_save + name_cap + "-12-F" + pic_type)
    fig_3.savefig(dir_save + name_cap + "-12-geom" + pic_type)

    for fig in [fig_1,fig_21,fig_22,fig_3]:
        fig.show()
