import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors

from vald_init import *

from pyMechOpt.multi_obj import mo_problem
from pyMechOpt.sim import spcs_name_idx, sim_1d, sim_0d_orig
from pyMechOpt.basic import x2yaml

mo_dir = "../moo/"
mech_dir = "../mech/"


def gen_best(name, name_cap):
    res_dir = mo_dir + "./res_" + name + "/"
    hist_dir = mo_dir + "./hist_" + name + "/"
    res_f, orig_f = mo_problem.load_res(
        res_dir=res_dir, hist_dir=hist_dir, load_orig=True
    )
    res_x = np.loadtxt(res_dir + "res_X.dat")
    t_f_normalization, t_f_norm = mo_problem.pareto_normalization(res_f)
    t_idx_min = np.argmin(t_f_norm)
    idx_min = t_idx_min
    f_min = t_f_norm[t_idx_min]
    print(t_idx_min)
    x_min = res_x[t_idx_min, :]

    gas_orig = ct.Solution(mech_dir + "gri30-r45.yaml")
    skip = np.loadtxt(res_dir + "skip.dat", dtype=bool)
    x2yaml(gas_orig, x_min, mech_dir + "mech.opt." + name + ".yaml", skip=skip)

    return


def vald_mo(name, name_cap, save_legend=False):
    hist_f, time = mo_problem.load_hist(hist_dir=mo_dir + "./hist_" + name + "/")
    res_f, orig_f = mo_problem.load_res(
        res_dir=mo_dir + "./res_" + name + "/",
        hist_dir=mo_dir + "./hist_" + name + "/",
        load_orig=True,
    )

    fig_1, t_lns = mo_problem.plot_hist(hist_f, markersize=1.5)
    for ax in fig_1.axes:
        ax.set_ylim(5e-05, 5)
        legend = ax.get_legend()

    if save_legend:
        plt.figure()
        new_fig, new_ax = plt.subplots(figsize=(6.5, 0.35))
        new_legend = new_ax.legend(
            handles=legend.legendHandles,
            labels=[l.get_text() for l in legend.get_texts()],
            loc="center",
            ncol=6,
            frameon=False,
        )
        new_ax.axis("off")
        new_fig.tight_layout()
        new_fig.savefig(dir_save + "hist-legend.pdf")

        plt.close(new_fig)

    legend.remove()

    fig_21, fig_22, t_lns_21, t_lns_22 = mo_problem.plot_parallel_coordinates(
        res_f, orig_f
    )
    for ax in fig_21.axes:
        ax.set_ylim(1e-07, 1)

    fig_3, t_lns, f_geom, f_mean, f_min, f_max = mo_problem.plot_hist_geom(hist_f)

    for fig in [fig_1, fig_21, fig_22, fig_3]:
        fig.set_size_inches(a, b)
        fig.tight_layout()

    fig_1.savefig(dir_save + name_cap + "-hist" + pic_type)
    fig_21.savefig(dir_save + name_cap + "-parallelFPreto" + pic_type)
    fig_22.savefig(dir_save + name_cap + "-parallelFNorm" + pic_type)
    fig_3.savefig(dir_save + name_cap + "-geom" + pic_type)

    for fig in [fig_1, fig_21, fig_22, fig_3]:
        fig.show()


def vald_delay(name, name_cap):
    mech_vald = mech_dir + "mech.opt." + name + ".yaml"
    pic_file = dir_save + name_cap + "-delay" + pic_type
    temp_ini = 1000
    pres = 1e05 * np.array([1, 5, 24, 100])
    ratio_eq = np.logspace(np.log(0.25), np.log(1 / 0.25), 9, base=np.e)
    vald = vald_mech(mech_dir + "gri30.yaml", mech_dir + "gri30-r45.yaml")
    error_rdct, delay_orig, delay_rdct = vald.err_delay_ratio_pres(
        temp_ini=temp_ini, ratio=ratio_eq, pres=pres
    )

    vald = vald_mech(mech_dir + "gri30.yaml", mech_vald)
    error_optd, delay_orig, delay_optd = vald.err_delay_ratio_pres(
        temp_ini=temp_ini, ratio=ratio_eq, pres=pres, delay_orig=delay_orig
    )

    color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    fig = plt.figure()
    for k in range(len(pres)):
        str_label = str(pres[k] / 1e05) + " bar" + " Unoptimized"
        plt.plot(
            ratio_eq,
            error_rdct[k][:],
            color[k] + "-s",
            label="{:>7}".format(str_label),
        )
        str_label = str(pres[k] / 1e05) + " bar" + " Optimized"
        plt.plot(
            ratio_eq,
            error_optd[k][:],
            color[k] + "--x",
            label="{:>7}".format(str_label),
        )
    plt.xlabel(r"$\mathrm{Ratio}$")
    plt.ylabel(r"$\mathrm{Error}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pic_file)
    plt.xscale("log")
    plt.show()


def vald_delay_single(name_list, name_cap_list):
    temp_ini = 1000
    pres = 1e05 * np.array([1, 10, 100])
    ratio_eq = np.logspace(np.log(0.25), np.log(1 / 0.25), 9, base=np.e)
    # color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    colors = plt.cm.tab20c(np.linspace(0, 1, 10))
    color_strings = [mcolors.rgb2hex(c) for c in colors]
    color_list = [f"#{c[1:7]}" for c in color_strings]
    print(color_list)
    for i in range(len(name_list)):
        name = name_list[i]
        name_cap = name_cap_list[i]
        mech_vald = mech_dir + name

        # alpha = np.linspace(0.25, 4, 16)
        # ratio_eq = 1 / alpha

        vald = vald_mech(mech_dir + "gri30.yaml", mech_vald)
        try:
            error_optd, delay_orig, delay_optd = vald.err_delay_ratio_pres(
                temp_ini=temp_ini, ratio=ratio_eq, pres=pres, delay_orig=delay_orig
            )
        except:
            error_optd, delay_orig, delay_optd = vald.err_delay_ratio_pres(
                temp_ini=temp_ini, ratio=ratio_eq, pres=pres
            )

        for k in range(len(pres)):
            plt.figure(k, figsize=(a, b))
            plt.plot(
                ratio_eq,
                error_optd[k],
                color=color_list[i],
                marker="s",
                label="{:>7}".format(name_cap),
            )
    xticks_positions = [0.25, 0.5, 1, 2, 4]
    xticks_labels = ["0.25", "0.5", "1", "2", "4"]
    for k in range(len(pres)):
        plt.figure(k)
        plt.xscale("log")
        plt.xticks(xticks_positions, xticks_labels)
        plt.xlabel(r"$\mathrm{Ratio}$")
        plt.ylabel(r"$\mathrm{Error}$")
        plt.legend(loc=1)
        plt.tight_layout()
        pic_file = dir_save + "delay-" + str(pres[k] / 1e05) + "bar" + pic_type
        plt.savefig(pic_file)
    plt.show()


def vald_parallel(name_list, name_cap_list):
    pressure = 1e05 * np.array([1, 1, 1, 100, 100, 100])
    temperature = np.array([1000, 1000, 1000, 1000, 1000, 1000])
    ratio = np.array([0.25, 1, 4, 0.25, 1, 4])
    n_obj = len(ratio)
    plt.figure(figsize=(a * 1.5, b))

    colors = plt.cm.tab20c(np.linspace(0, 1, 10))
    color_strings = [mcolors.rgb2hex(c) for c in colors]
    color_list = [f"#{c[1:7]}" for c in color_strings]
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
            "x-",
            color=color_list[k],
            markeredgecolor="k",
            label=r"$\mathrm{" + name_cap_list[k] + "}$",
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
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        fancybox=True,
        frameon=False,
        # borderaxespad=0.2
    )
    plt.ylabel(r"${F_{\mathrm{o}}}$")
    plt.xlabel(r"$\rm{OBJ}$")
    plt.tight_layout()
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5, rect=(0, 0, 1, 0.9))
    plt.savefig(dir_save + "f_mo_all" + pic_type)
    plt.show()


def vald_delay_gas_alpha(name_list, name_cap_list):
    temp_ini = 1000
    pres = 1e05
    # ratio_eq = np.logspace(0.4, 2.5, 16)
    ratio_eq = np.logspace(np.log(0.25), np.log(4), 16, base=np.e)

    t_pic_name = ""
    color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    k = 0
    for t_name in name_list:
        t_name = name_list[k]
        t_gas = ct.Solution(mech_dir + t_name)
        t_delay = np.zeros_like(ratio_eq)
        for l in range(len(ratio_eq)):
            t_gas.TP = temp_ini, pres
            t_gas.set_equivalence_ratio(ratio_eq[l], "CH4:1", "O2:1")
            t_delay[l] = sim_0d_delay(t_gas)
        plt.plot(ratio_eq, t_delay, color[k] + "-o", label=name_cap_list[k])
        k += 1
    plt.legend()
    plt.xscale("log")
    plt.show()


def vald_1D(name_list, name_cap_list):
    pres = 1e05
    temp_ini = 600
    ratio = 2
    spcs = ["CH4", "O2", "CH2O"]

    t_pic_name = ""
    color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    k = 0
    for t_name in name_list:
        t_name = name_list[k]
        t_gas = ct.Solution(mech_dir + t_name)
        t_spcs_idx = spcs_name_idx(t_gas, spcs)
        f = sim_1d(t_gas, temp_ini=temp_ini, pres_ini=pres, alpha=2.0, width=0.03)
        t_grid = f.grid
        t_T = f.T
        t_pic_name += name_cap_list[k] + "-"

        # plt.plot(t_grid, t_T, color[k], label=name_cap_list[k])
        for p in range(len(t_spcs_idx)):
            plt.plot(t_grid, f.X[t_spcs_idx[p]])
        k += 1
    plt.legend()
    plt.savefig(dir_save + "1D" + pic_type)
    plt.show()


def vald_0D(name_list, name_cap_list):
    pres = 1e05
    temp_ini = 1000
    ratio = 2
    # spcs = ["CH4", "O2", "CO", "CO2"]
    spcs = ["OH"]

    t_pic_name = ""
    color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    k = 0
    for t_name in name_list:
        t_name = name_list[k]
        t_gas = ct.Solution(mech_dir + t_name)
        t_gas.set_equivalence_ratio(ratio, "CH4:1", "O2:1")
        t_gas.TP = temp_ini, pres
        t_time, t_temp, t_y = sim_0d_orig(t_gas)
        t_spcs_idx = spcs_name_idx(t_gas, spcs)
        # plt.plot(t_grid, t_T, color[k], label=name_cap_list[k])
        for p in range(len(t_spcs_idx)):
            plt.plot(t_time, t_y[:, t_spcs_idx[p]])
        k += 1
    plt.legend()
    plt.savefig(dir_save + "0D" + pic_type)
    plt.show()


def vald_time(name, name_cap):
    pres = 1e05
    temp_ini = 1000
    ratio = 2
    spcs = ["CH4", "O2", "CO", "CO2"]
    gas_orig = ct.Solution(mech_dir + "gri30.yaml")
    gas_rdct = ct.Solution(mech_dir + "gri30-r45.yaml")
    gas_optd = ct.Solution(mech_dir + "mech.opt." + name + ".yaml")
