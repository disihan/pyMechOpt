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
# len_2=len(name_2)
f_list = []
time_list = []
color = ["r", "g", "y", "b", "c", "k", "m"]
for k in range(len_1):
    t_f, t_time = ls_problem.load_hist(hist_dir=so_dir + "hist_" + name_1[k] + "/")
    print(
        (Name_1[k] + "   F_Min:%.5e" + "    Time_avg:%.5f ")
        % (np.min(t_f[:1000]), t_time[999] / 1000)
    )
    t_f = t_f[:100]
    f_list.append(list(t_f))
    time_list.append(list(t_time))
    # plt.plot(
    #     np.arange(1, len(t_f) + 1, 1),
    #     t_f,
    #     color[k] + "--o",
    #     label=Name_1[k],
    #     markersize=4,
    # )

# plt.xlabel(r"$\mathrm{Iteration}$")
# plt.ylabel(r"$f$")
# plt.yscale("log")
# plt.grid()
# plt.legend(loc=0)
# plt.tight_layout()
# plt.savefig(dir_save + "cd_comp" + pic_type)
# plt.show()
