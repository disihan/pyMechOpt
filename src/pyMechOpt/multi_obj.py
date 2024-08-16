import numpy as np
import time
import cantera as ct
import sys
import os
import warnings
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
from pyMechOpt.basic import write_yaml, x2yaml


class mo_problem(basic_problem):
    """
    A class for chemcal reaction mechanisms based on multi-objective algorithm.
    The available algorithms are NSGA2, NSGA3, or other Pymoo provided algorithms.
    """

    def __init__(
        self,
        gas_orig,
        gas_rdct,
        fuel,
        oxydizer,
        temp_ini=np.array([800]),
        ratio=np.array([1]),
        pres=np.array([101325]),
        spcs_int=["H2"],
        spcs_peak=["OH"],
        range_input=0.2,
        *args,
        **kwargs,
    ):
        super().__init__(
            gas_orig,
            gas_rdct,
            fuel,
            oxydizer,
            temp_ini=temp_ini,
            ratio=ratio,
            pres=pres,
            spcs_int=spcs_int,
            spcs_peak=spcs_peak,
            *args,
            **kwargs,
        )
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
            (
                "iter: %d,    num:%d,    f:"
                + np.array2string(f, floatmode="fixed", precision=4)
            )
            % (self.gen, self.num)
        )
        self.hist_f.append(f)
        if len(self.hist_f) == self.pop_size:
            self.time_1 = time.process_time() - self.time_0
            self.gen = self.gen + 1
            np.savetxt(self.hist_dir + "gen_" + str(self.gen) + "_x.dat", self.hist_x)
            np.savetxt(self.hist_dir + "gen_" + str(self.gen) + "_f.dat", self.hist_f)
            self.hist_f = []
            self.hist_x = []
            with open(self.hist_dir + "time.dat", "a") as t_f:
                np.savetxt(t_f, np.array([self.gen, self.time_1]), newline=" ")
                t_f.write("\n")
            self.num = 0
        out["F"] = f

    def run(self, algorithm="NSGA3", max_gen=400, pop_size=200, **kwargs):
        super().moo_init(**kwargs)
        self.update_boundary()
        self.time_0 = time.process_time()
        if type(algorithm).__name__ == "str":
            if (
                algorithm == "NSGA3"
                or algorithm == "MOEAD"
                or algorithm == "CTAEA"
                or "UNSGA3"
            ):
                print("Calculating reference directions for " + algorithm + ".")
                ref_dirs = get_reference_directions("energy", self.nn, pop_size, seed=1)
                np.savetxt(self.res_dir + "ref_dirs.dat", ref_dirs)
                print("Done. ")
                if algorithm == "NSGA3":
                    algorithm = NSGA3(
                        pop_size=len(ref_dirs), ref_dirs=ref_dirs, **kwargs
                    )
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
        t_err_int, t_err_peak, t_err_temp_int, t_err_temp_peak = self.val_mo_err(
            np.zeros(self.n_var_s)
        )
        f_orig = self.err2mo(t_err_int, t_err_peak, t_err_temp_int, t_err_temp_peak)
        np.savetxt(self.hist_dir + "f_orig.dat", f_orig)
        t_k = 0
        for err_orig in [t_err_int, t_err_peak, t_err_temp_int, t_err_temp_peak]:
            t_k += 1
            np.savetxt(self.hist_dir + "err_orig" + str(t_k) + ".dat", err_orig)
        print("Pop size: %d" % (self.pop_size))
        print("START OPTIMIZATION.")
        res = minimize(self, algorithm, termination=("n_gen", max_gen), **kwargs)
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
    def gen_best(name, name_cap, mo_dir="../moo/", mech_dir="../mech/"):
        res_dir = mo_dir + "./res_" + name + "/"
        hist_dir = mo_dir + "./hist_" + name + "/"
        res_f, orig_f = mo_problem.load_res(
            res_dir=res_dir, hist_dir=hist_dir, load_orig=True
        )
        res_x = np.loadtxt(res_dir + "res_X.dat")
        t_f_normalization, t_f_norm = mo_problem.pareto_norm(res_f)
        t_idx_min = np.argmin(t_f_norm)
        idx_min = t_idx_min
        f_min = t_f_norm[t_idx_min]
        x_min = res_x[t_idx_min, :]

        gas_orig = ct.Solution(mech_dir + "gri30-r45.yaml")
        skip = np.loadtxt(res_dir + "skip.dat", dtype=bool)
        x2yaml(gas_orig, x_min, mech_dir + "mech.opt." + name + ".yaml", skip=skip)

        return

    @staticmethod
    def gen_ranking(res_dir, method, weights=None):
        res_f = mo_problem.load_res(res_dir=res_dir)
        n_obj = res_f.shape[1]
        res_x = np.loadtxt(res_dir + "res_X.dat")
        if weights is None:
            weights = np.ones(n_obj) * 1.0 / n_obj

        if method == "euclidean":
            _f_normalization, _f_so = mo_problem.pareto_norm(res_f)

        elif method == "sum":
            _f_normalization = mo_problem.pareto_normalization(res_f)
            _f_so = np.sum(_f_normalization, axis=1)

        elif method == "critic":
            _f_normalization = mo_problem.pareto_normalization(res_f)
            _fn_mean = np.mean(_f_normalization, axis=0)
            _s_fn = np.std(_f_normalization, ddof=1, axis=0)
            _r_fn = np.corrcoef(_f_normalization.transpose())
            _rr_fn = np.sum(1 - _r_fn, axis=0)
            _c = _s_fn * _rr_fn
            _sum_c = np.sum(_c)
            wt = _c / _sum_c
            _f_so = np.einsum("j,ij->i", wt, _f_normalization)

        elif method == "topsis":
            from pyDecision.algorithm import topsis_method

            _f_normalization = mo_problem.pareto_normalization(res_f)
            criterion_type = ["min"] * n_obj
            _f_so = topsis_method(
                _f_normalization,
                weights,
                criterion_type,
                graph=False,
                verbose=False,
            )
            _f_so = 1 - _f_so

        # elif method == "entropy":
        #     from pyDecision.algorithm import entropy_method

        #     _f_normalization = mo_problem.pareto_normalization(res_f)
        #     criterion_type = ["min"] * n_obj
        #     wt = entropy_method(_f_normalization, criterion_type)

        #     _f_so = np.einsum("j,ij->i", wt, _f_normalization)

        else:
            warnings.warn("Unsupported method  " + method + ".")
            exit()
        rk = np.argsort(_f_so)
        return (
            _f_so,
            rk,
        )

    @staticmethod
    def load_hist(hist_dir="./hist/"):
        k = 1
        hist_f = []
        while True:
            t_fname = "gen_" + str(k) + "_f.dat"
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
    def plot_hist(hist_f, errorbar=False, **kwargs):
        fig = plt.figure()
        f_min = []
        f_max = []
        f_mean = []
        f_err = []
        color = ["r", "g", "y", "b", "c", "k", "m"]
        marker = ["o", "v", "*", "x", "s", "d"]
        ms = kwargs.get("markersize", 10)
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
        ln_style = "--"
        if len(f_mean) > 50:
            marker = [""] * len(marker)
            ln_style = "-"
        if errorbar:
            for k in range(n_obj):
                style_str = ln_style + marker[k % len(marker)] + color[k % len(color)]
                lns += plt.errorbar(
                    np.arange(1, len(hist_f) + 1),
                    f_mean[:, k],
                    err=f_err,
                    fmt=style_str,
                    label=r"$\rm{OBJ\ " + str(k + 1) + "}$",
                    markersize=ms,
                )
        else:
            for k in range(n_obj):
                style_str = ln_style + marker[k % len(marker)] + color[k % len(color)]
                lns += plt.errorbar(
                    np.arange(1, len(hist_f) + 1),
                    f_mean[:, k],
                    fmt=style_str,
                    label=r"$\rm{OBJ " + str(k + 1) + "}$",
                    markersize=ms,
                )

        plt.xlabel("$\mathrm{Interation}$")
        plt.ylabel("$\overline{F}$")
        plt.yscale("log")
        if n_obj < 7:
            plt.legend(loc="upper right")
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

        plt.fill_between(
            np.arange(1, len(hist_f) + 1), f_min, f_max, alpha=0.3, facecolor="gray"
        )
        lns += plt.plot(np.arange(1, len(hist_f) + 1), f_mean, "r--*")
        plt.yscale("log")
        plt.grid()
        plt.ylabel("${F_{\mathrm{GEOM}}}$")
        plt.xlabel("$\mathrm{Interation}$")
        plt.tight_layout()

        return fig, lns, f_geom, f_mean, f_min, f_max

    @staticmethod
    def plot_hist_norm(hist_f):
        f_geom = []
        f_mean = []
        f_min = []
        idx_min = []
        f_max = []
        fig = plt.figure()
        for k in range(len(hist_f)):
            t_f_geom = np.linalg.norm(hist_f[k], axis=1)
            f_geom.append(t_f_geom)
            f_mean.append(np.mean(t_f_geom))
            t_idx_min = np.argmin(t_f_geom)
            idx_min.append(t_idx_min)
            f_min.append(t_f_geom[t_idx_min])
            f_max.append(np.max(t_f_geom))

        lns = []

        plt.fill_between(
            np.arange(1, len(hist_f) + 1), f_min, f_max, alpha=0.3, facecolor="gray"
        )
        lns += plt.plot(np.arange(1, len(hist_f) + 1), f_mean, "r--*")
        plt.yscale("log")
        plt.grid()
        plt.ylabel("${F_{\mathrm{GEOM}}}$")
        plt.xlabel("$\mathrm{Interation}$")
        plt.tight_layout()

        return fig, lns, f_geom, f_mean, f_min, f_max, idx_min

    @staticmethod
    def pareto_norm(res_f):
        F_imin_1 = np.expand_dims(np.min(res_f, axis=0), 0)
        F_imax_1 = np.expand_dims(np.max(res_f, axis=0), 0)
        F_imin = F_imin_1.repeat(res_f.shape[0], axis=0)
        F_imax = F_imax_1.repeat(res_f.shape[0], axis=0)
        F_o = (res_f - F_imin) / (F_imax - F_imin)
        F_o_norm = np.linalg.norm(F_o, ord=2, axis=1)
        return F_o, F_o_norm

    @staticmethod
    def pareto_normalization(res_f):
        F_imin_1 = np.expand_dims(np.min(res_f, axis=0), 0)
        F_imax_1 = np.expand_dims(np.max(res_f, axis=0), 0)
        F_imin = F_imin_1.repeat(res_f.shape[0], axis=0)
        F_imax = F_imax_1.repeat(res_f.shape[0], axis=0)
        F_o = (res_f - F_imin) / (F_imax - F_imin)
        return F_o

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
        color = ["r", "g", "y", "b", "c", "k", "m"]
        opt_ratio_mean = np.mean(opt_ratio, axis=0)
        t_c = "lightgrey"
        lns_1 = []
        lns_2 = []
        for k in range(opt_ratio.shape[0]):
            if highlight:
                if (opt_ratio[k, :] < opt_ratio_mean).all():
                    t_c = "lightgrey"
                else:
                    t_c = "lightgrey"
            plt.figure(num_1)
            lns_1 += plt.plot(
                np.linspace(1, n_obj, n_obj),
                res_f[k, :],
                "x--",
                color=t_c,
                markeredgecolor=color[k % (len(color))],
            )
            plt.figure(num_2)
            lns_2 += plt.plot(
                np.linspace(1, n_obj, n_obj),
                F_o[k, :],
                "x--",
                color=t_c,
                markeredgecolor=color[k % (len(color))],
            )
        sct_x = []
        for k in range(n_obj):
            # sct_x.append("OBJ " + str(k + 1))
            sct_x.append(str(k + 1))
        plt.figure(num_1)
        plt.grid()
        plt.xticks(np.linspace(1, n_obj, n_obj), labels=sct_x)
        plt.yscale("log")
        plt.ylabel(r"${F_{\mathrm{parato}}}$")
        plt.xlabel(r"$\rm{OBJ}$")
        plt.tight_layout()

        plt.figure(num_2)
        plt.xticks(np.linspace(1, n_obj, n_obj), labels=sct_x)
        plt.ylabel(r"$\tilde{F}$")
        plt.xlabel(r"$\rm{OBJ}$")
        plt.tight_layout()

        return fig_1, fig_2, lns_1, lns_2

    def single_parallel_coordinates(
        self,
    ):
        return self.val_mo(np.zeros(self.n_var_s))
