import numpy as np
import time
import cantera as ct
import sys
import os
import matplotlib.pyplot as plt

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
# from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.optimize import minimize

from pyMechOpt.basic import basic_problem
from pyMechOpt.basic import write_yaml


class mo_problem(basic_problem):
    """
    A class for chemcal reaction mechanisms based on multi-objective algorithm.
    The available algorithms are NSGA2, NSGA3, or other Pymoo provided algorithms.
    """

    def __init__(self, gas_orig, gas_rdct, temp_ini=np.array([800]), ratio=np.array([1]), pres=np.array([101325]),
                 spcs_int=["H2"], spcs_peak=["OH"], range_input=0.2, *args, **kwargs):
        super().__init__(gas_orig, gas_rdct, temp_ini=temp_ini, ratio=ratio, pres=pres,
                         spcs_int=spcs_int, spcs_peak=spcs_peak, *args, **kwargs)
        self.range_input = range_input
        self.l_s = (-1) * np.ones(self.n_var_o) * range_input
        self.r_s = np.ones(self.n_var_o) * range_input
        return

    def out_init(self):
        super().out_init()
        if os.path.exists(self.hist_dir + "time.dat") == True:
            os.remove(self.hist_dir + "time.dat")
        return

    def _evaluate(self, x, out, *args, **kwargs):
        self.hist_x.append(x)
        f = self.val_mo(x)
        self.num += 1
        print(
            ("iter: %d,    num:%d,    f:" + np.array2string(f, floatmode="fixed", precision=4)) % (self.gen, self.num))
        self.hist_f.append(f)
        if len(self.hist_f) == self.pop_size:
            self.time_1 = time.process_time() - self.time_0
            self.gen = self.gen + 1
            np.savetxt(self.hist_dir + 'gen_' + str(self.gen) + '_x.dat', self.hist_x)
            np.savetxt(self.hist_dir + 'gen_' + str(self.gen) + '_f.dat', self.hist_f)
            self.hist_f = []
            self.hist_x = []
            with open(self.hist_dir + "time.dat", "a") as t_f:
                np.savetxt(t_f, np.array([self.gen, self.time_1]), newline=' ')
                t_f.write("\n")
            self.num = 0
        out["F"] = f

    def run(self, algorithm="NSGA3", max_gen=400, pop_size=200, **kwargs):
        super().moo_init(**kwargs)
        self.update_boundary()
        self.time_0 = time.process_time()
        if type(algorithm).__name__ == "str":
            if algorithm == "NSGA3" or algorithm == "MOEAD" or algorithm == "CTAEA" or "UNSGA3":
                print("Calculating reference directions for " + algorithm + ".")
                ref_dirs = get_reference_directions("energy", self.nn, pop_size, seed=1)
                np.savetxt(self.res_dir + "ref_dirs.dat", ref_dirs)
                print("Done. ")
                if algorithm == "NSGA3":
                    algorithm = NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs, **kwargs)
                elif algorithm == "MOEAD":
                    algorithm = MOEAD(ref_dirs=ref_dirs, **kwargs)
                elif algorithm == "CTAEA":
                    algorithm = CTAEA(ref_dirs=ref_dirs, **kwargs)
                elif algorithm == "UNSGA3":
                    algorithm = UNSGA3(ref_dirs=ref_dirs, **kwargs)
            elif algorithm == "NSGA2":
                algorithm = NSGA2(pop_size=pop_size, **kwargs)
            else:
                sys.stderr.write("Error occurred: Unsupported algorithm.\n")
                exit()

        self.pop_size = algorithm.pop_size
        f_orig = self.val_mo(np.zeros(self.n_var_s))
        np.savetxt(self.hist_dir + "f_orig.dat", f_orig)
        print("Pop size: %d" % (self.pop_size))
        print("START OPTIMIZATION.")
        res = minimize(self, algorithm, termination=('n_gen', max_gen), **kwargs)
        print("DONE OPTIMIZATION")
        print("The total number of solutions in the Pareto front: %d" % len(res.F))
        np.savetxt(self.res_dir + "res_X.dat", res.X)
        np.savetxt(self.res_dir + "res_F.dat", res.F)
        for k in range(len(res.F)):
            t_fname = self.mech_res + str(k + 1) + ".yaml"
            print("Writing result yaml file: " + t_fname)
            t_X = res.X[k, :]
            t_gas_modd = self.input2gas(t_X)
            write_yaml(t_gas_modd, self.res_dir + t_fname)
        return res

    @staticmethod
    def load_hist(hist_dir="./hist/"):
        k = 1
        hist_f = []
        while True:
            t_fname = 'gen_' + str(k) + '_f.dat'
            t_fname = hist_dir + t_fname
            if os.path.exists(t_fname) == True:
                print("Loading history data: " + t_fname)
                t_f = np.loadtxt(t_fname)
                hist_f.append(t_f)
            else:
                break
            k = k + 1
        time = np.loadtxt(hist_dir + "time.dat")
        return hist_f, time

    @staticmethod
    def load_res(hist_dir="./hist/", res_dir="./res/", load_orig=False):
        res_f = np.loadtxt(res_dir + "res_F.dat")
        if load_orig:
            orig_f = np.loadtxt(hist_dir + "f_orig.dat")
            return res_f, orig_f
        return res_f

    @staticmethod
    def plot_hist(hist_f, errorbar=False):
        fig = plt.figure()
        f_min = []
        f_max = []
        f_mean = []
        f_err = []
        color = ['r', 'g', 'y', 'b', 'c', 'k', 'm']
        marker = ['o', 'v', '*', 'x', 's', 'd']
        for k in range(len(hist_f)):
            t_f = hist_f[k]
            f_min.append(np.min(t_f, axis=0))
            f_max.append(np.max(t_f, axis=0))
            f_mean.append(np.mean(t_f, axis=0))
            f_err.append(np.std(t_f, axis=0))

        f_min = np.array(f_min)
        f_max = np.array(f_max)
        f_mean = np.array(f_mean)
        f_err = np.array(f_err)

        n_obj = t_f.shape[1]
        if n_obj > len(color) * len(marker):
            sys.stderr.write("Warning: n_obj>" + str(len(color) * len(marker)))
        lns = []
        if errorbar:
            for k in range(n_obj):
                style_str = "--" + marker[k % len(marker)] + color[k % len(color)]
                lns += plt.errorbar(np.arange(1, len(hist_f) + 1), f_mean[:, k], err=f_err, fmt=style_str,
                                    label=r'${OBJ}_{' + str(k + 1) + '}$')
        else:
            for k in range(n_obj):
                style_str = "--" + marker[k % len(marker)] + color[k % len(color)]
                lns += plt.errorbar(np.arange(1, len(hist_f) + 1), f_mean[:, k], fmt=style_str,
                                    label=r'${OBJ}_{' + str(k + 1) + '}$')

        plt.xlabel("$\mathrm{Interation}$")
        plt.ylabel("$\overline{F}$")
        plt.yscale("log")
        if n_obj < 7:
            plt.legend()
        plt.grid()
        plt.tight_layout()
        return fig, lns

    @staticmethod
    def plot_hist_geom(hist_f):
        f_geom = []
        f_mean = []
        f_min = []
        f_max = []
        fig = plt.figure()
        for k in range(len(hist_f)):
            t_f_geom = np.exp(np.log(hist_f[k]).mean(axis=1))
            f_geom.append(t_f_geom)
            f_mean.append(np.mean(t_f_geom))
            f_min.append(np.min(t_f_geom))
            f_max.append(np.max(t_f_geom))

        lns = []

        plt.fill_between(np.arange(1, len(hist_f) + 1), f_min, f_max, alpha=0.3, facecolor='gray')
        lns += plt.plot(np.arange(1, len(hist_f) + 1), f_mean, 'r--*')
        plt.yscale("log")
        plt.grid()
        plt.ylabel("${F_{\mathrm{GEOM}}}$")
        plt.xlabel("$\mathrm{Interation}$")
        plt.tight_layout()

        return fig, lns, f_geom, f_mean, f_min, f_max

    @staticmethod
    def plot_hist_rms(hist_f):
        f_rms = []
        f_mean = []
        f_min = []
        f_max = []
        fig = plt.figure()
        for k in range(len(hist_f)):
            t_f_rms = np.linalg.norm(hist_f[k], axis=1)
            f_rms.append(t_f_rms)
            f_mean.append(np.mean(t_f_rms))
            f_min.append(np.min(t_f_rms))
            f_max.append(np.max(t_f_rms))

        lns = []

        plt.fill_between(np.arange(1, len(hist_f) + 1), f_min, f_max, alpha=0.3, facecolor='gray')
        lns += plt.plot(np.arange(1, len(hist_f) + 1), f_mean, 'r--*')
        plt.yscale("log")
        plt.grid()
        plt.ylabel(r"${F_{\mathrm{RMS}}}$")
        plt.xlabel("$\mathrm{Interation}$")
        plt.xlabel()
        plt.tight_layout()

        return fig, lns, f_rms, f_mean, f_min, f_max

    @staticmethod
    def plot_parallel_coordinates(res_f, orig_f, highlight=False, **kwargs):
        fig_1 = plt.figure(**kwargs)
        fig_2 = plt.figure(**kwargs)
        num_1 = fig_1.number
        num_2 = fig_2.number
        n_obj = res_f.shape[1]
        F_imin_1 = np.expand_dims(np.min(res_f, axis=0), 0)
        F_imax_1 = np.expand_dims(np.max(res_f, axis=0), 0)
        F_imin = F_imin_1.repeat(res_f.shape[0], axis=0)
        F_imax = F_imax_1.repeat(res_f.shape[0], axis=0)
        F_o = (res_f - F_imin) / (F_imax - F_imin)
        # F_orig = np.loadtxt("./hist/f_orig.dat")
        F_orig = np.expand_dims(orig_f, 0).repeat(res_f.shape[0], axis=0)
        opt_ratio = res_f / F_orig
        color = ['r', 'g', 'y', 'b', 'c', 'k', 'm']
        opt_ratio_mean = np.mean(opt_ratio, axis=0)
        t_c = 'lightgrey'
        lns_1 = []
        lns_2 = []
        for k in range(opt_ratio.shape[0]):
            if highlight:
                if (opt_ratio[k, :] < opt_ratio_mean).all():
                    t_c = 'lightgrey'
                else:
                    t_c = 'lightgrey'
            plt.figure(num_1)
            lns_1 += plt.plot(np.linspace(1, n_obj, n_obj), opt_ratio[k, :], 'x--', color=t_c,
                              markeredgecolor=color[k % (len(color))])
            plt.figure(num_2)
            lns_2 += plt.plot(np.linspace(1, n_obj, n_obj), F_o[k, :], 'x--', color=t_c,
                              markeredgecolor=color[k % (len(color))])
        sct_x = []
        for k in range(n_obj):
            # sct_x.append("OBJ " + str(k + 1))
            sct_x.append(str(k + 1))
        plt.figure(num_1)
        plt.grid()
        plt.xticks(np.linspace(1, n_obj, n_obj), labels=sct_x)
        plt.yscale("log")
        plt.ylabel(r"${F_{\mathrm{parato}}}/{F_0}$")
        plt.xlabel(r"$\rm{OBJ}$")
        plt.tight_layout()

        plt.figure(num_2)
        plt.xticks(np.linspace(1, n_obj, n_obj), labels=sct_x)
        plt.ylabel(r"$\tilde{F}$")
        plt.xlabel(r"$\rm{OBJ}$")
        plt.tight_layout()

        return fig_1, fig_2, lns_1, lns_2
