# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:08:11 2023

@author: DSH
"""

import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

from pyMechOpt.line_search import ls_problem
from pyMechOpt.single_obj import so_problem
import vald_init
from vald_init import *

so_dir = "../ls and soo/"

name_1 = ["cd", "cd_sa", "cd_gss", "cd_sa_gss"]
Name_1 = ["CD_1", "CD_2", "CD_3", "CD_4"]

plt.figure(figsize=(a, b))
len_1 = len(name_1)
f_list = []
time_list = []
color = ["#FC757B", "#76CBB4", "#EECA40", "#3C9BC9", "#F88455", "#FDCA93", "#444444"]
for k in range(len_1):
    t_f, t_time = ls_problem.load_hist(hist_dir=so_dir + "hist_" + name_1[k] + "/")
    t_f = t_f[:1000]
    f_list.append(list(t_f))
    time_list.append(list(t_time))
    plt.plot(
        np.arange(1, len(t_f) + 1, 1),
        t_f,
        color=color[k],
        label=Name_1[k],
        linewidth=2,
    )


plt.xlabel(r"$\mathrm{Iteration}$")
plt.ylabel(r"$f$")
plt.yscale("log")
plt.grid(color="k", alpha=0.15)
plt.legend(fancybox=False, loc=1, ncol=2, edgecolor="#999999", framealpha=1)
plt.savefig(dir_save + "cd_comp" + pic_type)
plt.show()
