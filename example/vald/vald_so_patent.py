# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:08:11 2023

@author: DSH
"""

import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

# import sympy

from pyMechOpt.line_search import ls_problem
from pyMechOpt.single_obj import so_problem

# import vald_init
# from vald_init import *
pic_type = ".pdf"
dir_save = "../figures/"
mech_dir = "../mech/"

so_dir = "../ls and soo/"

plt.figure(figsize=[4.5, 3.5])

name_1 = ["gd", "cd"]
Name_1 = ["GD", "CD"]
name_2 = ["ga", "de"]
Name_2 = ["GA", "DE"]
len_1 = len(name_1)
len_2 = len(name_2)
f_list = []
fmin_list = []
time_list = []
color = ["r", "g", "y", "b", "c", "k", "m"]
ls = ["--", "-", "-.", ":"]
# marker = ["x", ""]
for k in range(len_1):
    t_f, t_time = ls_problem.load_hist(hist_dir=so_dir + "hist_" + name_1[k] + "/")
    f_list.append(list(t_f))
    fmin_list.append(np.min(t_f))
    time_list.append(list(t_time))
    plt.plot(t_time, t_f, "k" + ls[k], label=Name_1[k], markersize=3)

for k in range(len_2):
    t_f_all, t_f, t_time = so_problem.load_hist(
        hist_dir=so_dir + "hist_" + name_2[k] + "/"
    )
    f_list.append(list(t_f))
    fmin_list.append(np.min(t_f))
    time_list.append(list(t_time))
    plt.plot(t_time, t_f, "k" + ls[k + len_1], label=Name_2[k], markersize=3)
#
# plt.xscale("symlog")
plt.xlim([-50, 1000])
plt.yscale("log")
plt.grid()
plt.legend(loc=0)
plt.xlabel("$\mathrm{Execution\ time(s)}$")
plt.ylabel("$f$")
plt.tight_layout()
plt.savefig(dir_save + "iter_all_patent" + pic_type)

a = 4.5
b = 3.5

from pyMechOpt.multi_obj import mo_problem


def vald_parallel(name_list, name_cap_list):
    pressure = 1e05 * np.array([1, 1, 1, 100, 100, 100])
    temperature = np.array([1000, 1000, 1000, 1000, 1000, 1000])
    ratio = np.array([0.25, 1, 4, 0.25, 1, 4])
    n_obj = len(ratio)
    plt.figure(figsize=(a * 1.5, b))
    marker_2 = ["x", "o", "s", "^", "D"]

    spcs_int = ["CH4", "O2"]
    spcs_peak = ["CH2O", "CO", "CO2"]
    for k in range(len(name_list)):
        temp_opt_prob = mo_problem(
            "gri30.yaml",
            mech_dir + name_list[k],
            temperature,
            ratio,
            pressure,
            spcs_int,
            spcs_peak,
            hist_dir="./temp_hist_vald/",
            res_dir="./temp_res_vald/",
        )
        mo_res = temp_opt_prob.single_parallel_coordinates()
        plt.plot(
            np.linspace(1, n_obj, n_obj),
            mo_res,
            "k--" + marker_2[k],
            # markeredgecolor="k",
            label=r"$\mathrm{" + name_cap_list[k] + "}$",
            markersize=5,
        )
    sct_x = []
    for k in range(n_obj):
        # sct_x.append("OBJ " + str(k + 1))
        sct_x.append(str(k + 1))
    plt.grid()
    plt.xticks(np.linspace(1, n_obj, n_obj), labels=sct_x)
    plt.yscale("log")
    # plt.legend(loc=1, bbox_to_anchor=(1.8, 0.5))
    plt.legend(
        loc="upper center",
        # bbox_to_anchor=(0.5, -0.2),
        ncol=5,
        fancybox=True,
        frameon=False,
        # borderaxespad=0.2
    )
    plt.ylabel(r"${F_{\mathrm{m}}}$")
    plt.xlabel(r"$\rm{OBJ}$")
    plt.tight_layout()
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5, rect=(0, 0, 1, 0.9))
    plt.savefig(dir_save + "f_mo_all_patent" + pic_type)
    plt.show()


vald_parallel(
    [
        "gri30-r45.yaml",
        "mech.opt.gd.yaml",
        "mech.opt.cd.yaml",
        "mech.opt.ga.yaml",
        "mech.opt.de.yaml",
    ],
    [
        "R45",
        "GD",
        "CD",
        "GA",
        "DE",
    ],
)
