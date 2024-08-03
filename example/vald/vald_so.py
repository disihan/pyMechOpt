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

import vald_init
from vald_init import *

so_dir = "../ls and soo/"

plt.figure(figsize=[4.5, 3.5])

name_1 = ["gd", "cd", "sgd"]
Name_1 = ["GD", "CD", "SGD"]
name_2 = ["ga", "de", "es"]
Name_2 = ["GA", "DE", "ES"]
len_1 = len(name_1)
len_2 = len(name_2)
f_list = []
fmin_list = []
time_list = []
color = ["r", "g", "y", "b", "c", "k", "m"]
for k in range(len_1):
    t_f, t_time = ls_problem.load_hist(hist_dir=so_dir + "hist_" + name_1[k] + "/")
    f_list.append(list(t_f))
    fmin_list.append(np.min(t_f))
    time_list.append(list(t_time))
    plt.plot(t_time, t_f, color[k] + "--o", label=Name_1[k], markersize=3)

for k in range(len_2):
    t_f_all, t_f, t_time = so_problem.load_hist(
        hist_dir=so_dir + "hist_" + name_2[k] + "/"
    )
    f_list.append(list(t_f))
    fmin_list.append(np.min(t_f))
    time_list.append(list(t_time))
    plt.plot(t_time, t_f, color[len_1 + k] + "--s", label=Name_2[k], markersize=3)
#
# plt.xscale("symlog")
plt.xlim([-50, 1000])
plt.yscale("log")
plt.grid()
plt.legend(loc=0)
plt.xlabel("$\mathrm{Execution\ time(s)}$")
plt.ylabel("$f$")
plt.tight_layout()
plt.savefig(dir_save + "iter_all" + pic_type)

plt.figure(figsize=[4.5, 3.5])
c_1 = ["r"] * len(name_1)
c_2 = ["b"] * len(name_2)
plt.bar(np.arange(len(fmin_list)), height=np.array(fmin_list), color=c_1 + c_2)
k = 0
for fmin in fmin_list:
    plt.annotate("%.2E" % fmin, (k, fmin), ha="center", va="bottom")
    k += 1
plt.savefig(dir_save + "bar_all" + pic_type)
plt.yscale("log")
plt.ylabel("$f_{\mathrm{final}}$")
plt.xticks(np.arange(len(Name_1) + len(Name_2)), labels=Name_1 + Name_2, rotation=90)
plt.tight_layout()
plt.savefig(dir_save + "bar_all" + pic_type)
plt.show()
