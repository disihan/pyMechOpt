import numpy as np
import cantera as ct
import os
import sys
import time
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem

from pyMechOpt.sim import (
    calc_all_orig,
    calc_all_rdct,
    spcs_name_idx,
    calc_y_int,
    calc_temp_int,
    calc_err_all_list,
)
from pyMechOpt.mod_rxns import get_factor_dim, rxns_yaml_arr_list


class basic_problem(ElementwiseProblem):
    def __init__(
        self,
        gas_orig,
        gas_rdct,
        temp_ini=np.array([800]),
        ratio=np.array([1]),
        pres=np.array([101325]),
        spcs_int=["H2"],
        spcs_peak=["OH"],
        **kwargs,
    ):
        # global num, gen, total_num
        if type(gas_orig).__name__ == "str":
            gas_orig = ct.Solution(gas_orig)
        if type(gas_rdct).__name__ == "str":
            gas_rdct = ct.Solution(gas_rdct)

        self.gas_rdct = gas_rdct
        if (temp_ini.shape == pres.shape and pres.shape == ratio.shape) == False:
            sys.stderr.write(
                "Error occurred: The dimensions of the initial condition vectors for pressure, temperature, and mixing ratio must be equal.\n"
            )
            exit()
        self.ratio = ratio
        self.pres = pres
        self.temp_ini = temp_ini
        self.nn = len(temp_ini)

        self.idx_int_orig = spcs_name_idx(gas_orig, spcs_int)
        self.idx_int_rdct = spcs_name_idx(gas_rdct, spcs_int)
        self.idx_peak_orig = spcs_name_idx(gas_orig, spcs_peak)
        self.idx_peak_rdct = spcs_name_idx(gas_rdct, spcs_peak)
        # weight = ([1, 1, 1, 1],)
        # correct_peak = (False,)
        # self.weight = weight
        # self.norm = kwargs.get("norm", 2.0)
        # self.norm_int_int = kwargs.get("norm_int_int", True)
        # self.correct_peak = correct_peak
        self.kwargs = kwargs

        self.n_var_o = get_factor_dim(gas_rdct)
        self.n_var_s = self.n_var_o
        self.skip = np.zeros(self.n_var_o, dtype=bool)
        (
            self.t_list_orig,
            self.temp_list_orig,
            self.y_list_int_orig,
            self.y_list_peak_orig,
        ) = calc_all_orig(
            gas_orig, ratio, temp_ini, pres, self.idx_int_orig, self.idx_peak_orig
        )
        self.y_int_orig = calc_y_int(self.t_list_orig, self.y_list_int_orig)
        self.temp_int_orig = calc_temp_int(self.t_list_orig, self.temp_list_orig)

        self.hist_f = []
        self.hist_x = []
        self.num = 0
        self.gen = 0
        self.pop_size = 0
        self.hist_dir = kwargs.get("hist_dir", "./hist/")
        self.res_dir = kwargs.get("res_dir", "./res/")
        self.mech_res = "mech.opt."
        print("history dir: " + self.hist_dir)
        print("result dir: " + self.res_dir)
        self.out_init()
        return

    def input2gas(self, input):
        input_s = np.zeros(self.n_var_o)
        input_s[~self.skip] = input
        fac = input_s + 1
        gas_modd = rxns_yaml_arr_list(self.gas_rdct, fac)
        return gas_modd

    def val_mo_err(self, input):
        gas_modd = self.input2gas(input)
        t_list_modd, temp_list_modd, y_list_int_modd, y_list_peak_modd = calc_all_rdct(
            gas_modd,
            self.ratio,
            self.temp_ini,
            self.pres,
            self.idx_int_rdct,
            self.idx_peak_rdct,
            self.t_list_orig,
        )
        err_int, err_peak, err_temp_int, err_temp_peak = calc_err_all_list(
            self.t_list_orig,
            self.temp_list_orig,
            self.y_list_int_orig,
            self.y_list_peak_orig,
            t_list_modd,
            temp_list_modd,
            y_list_int_modd,
            y_list_peak_modd,
            self.y_int_orig,
            self.temp_int_orig,
            **self.kwargs,
        )
        err_int = np.array(err_int)
        err_peak = np.array(err_peak)
        err_temp_int = np.array(err_temp_int)
        err_temp_peak = np.array(err_temp_peak)
        err_int = np.sum(err_int, axis=1)
        err_peak = np.sum(err_peak, axis=1)
        # print("err_int:" + str(err_int))
        # print("err_peak:" + str(err_peak))
        # print("err_temp_int:" + str(err_temp_int))
        # print("err_temp_peak:" + str(err_temp_peak))
        return err_int, err_peak, err_temp_int, err_temp_peak

    def err2mo(self, err_int, err_peak, err_temp_int, err_temp_peak):
        weight = self.kwargs.get("weight", [1, 1, 1, 1])
        err = (
            weight[0] * err_int
            + weight[1] * err_peak
            + weight[2] * err_temp_int
            + weight[3] * err_temp_peak
        )
        return err

    def val_mo(self, input):
        err_int, err_peak, err_temp_int, err_temp_peak = self.val_mo_err(input)
        return self.err2mo(err_int, err_peak, err_temp_int, err_temp_peak)

    def save_f_orig_so(self):
        f_orig = self.val_so(np.zeros(self.n_var_s))
        np.savetxt(self.hist_dir + "f_orig.dat", [f_orig])
        return f_orig

    def val_so(self, input):
        err = self.val_mo(input)
        err = np.linalg.norm(err)
        return err

    def calc_skip(self, skip_num=None):
        print("Calculating the number of variables that can be skipped.")
        input = np.zeros(self.n_var_o)
        job = self.grad(input, grad_fac=0.01, log=True, mo=True)
        job_abs = np.abs(job)
        job_abs = np.linalg.norm(job_abs, axis=1)
        self.skip[job_abs == 0] = True

        nvar_zero = np.sum(self.skip)
        order = np.argsort(job_abs)
        ign_len = int((len(order) - nvar_zero) / 2)
        if skip_num is None:
            self.skip[order[: ign_len + nvar_zero]] = True
        else:
            self.skip[order[:skip_num]] = True

        np.savetxt(self.res_dir + "skip.dat", self.skip, fmt="%d")
        np.savetxt(self.res_dir + "order.dat", order, fmt="%d")
        np.savetxt(self.res_dir + "grad.dat", job)
        self.n_var_s = self.n_var_o - np.sum(
            np.asarray(self.skip[: self.n_var_o], dtype=int)
        )

        # self.load_skip()
        self.l_s = (-1) * np.ones(self.n_var_s) * self.range_input
        self.r_s = np.ones(self.n_var_s) * self.range_input
        print(
            "Done. total skip number: %d, n_var: %d" % (np.sum(self.skip), self.n_var_s)
        )
        return

    def update_boundary(self):
        self.l_s = (-1) * np.ones(self.n_var_s) * self.range_input
        self.r_s = np.ones(self.n_var_s) * self.range_input
        return

    def load_skip(self):
        print("Loading skip.dat.")
        file = self.res_dir + "skip.dat"
        self.skip = np.loadtxt(file, dtype=bool)
        self.n_var_s = self.n_var_o - np.sum(
            np.asarray(self.skip[: self.n_var_o], dtype=int)
        )
        self.l_s = (-1) * np.ones(self.n_var_s) * self.range_input
        self.r_s = np.ones(self.n_var_s) * self.range_input
        print("Done. total number: %d." % (np.sum(self.skip)))

    def grad(self, input, mo=False, grad_fac=1e-04, log=False):
        print("Calculating Jacobian matrix.")
        if mo:
            res = np.zeros([self.n_var_s, self.nn])
        else:
            res = np.zeros(self.n_var_s)

        for k in range(self.n_var_s):
            t_fac_1 = input.copy()
            t_fac_2 = input.copy()
            t_fac_1[k] = t_fac_1[k] + grad_fac
            t_fac_2[k] = t_fac_2[k] - grad_fac
            if mo:
                t_j = self.val_mo(t_fac_1) - self.val_mo(t_fac_2)
                t_res = t_j / grad_fac
                res[k, :] = t_res
            else:
                t_j = self.val_so(t_fac_1) - self.val_so(t_fac_2)
                t_res = t_j / grad_fac
                res[k] = t_res
            if log:
                print(("%d: " + str(t_res)) % (k))

        return res

    def out_init(self):
        if os.path.exists(self.hist_dir) == False:
            os.mkdir(self.hist_dir)
        if os.path.exists(self.res_dir) == False:
            os.mkdir(self.res_dir)
        return

    def soo_init(self, **kwargs):
        self.out_init()
        super().__init__(
            n_var=self.n_var_s, n_obj=1, n_constr=0, xl=self.l_s, xu=self.r_s, **kwargs
        )
        return

    def moo_init(self, **kwargs):
        self.out_init()
        super().__init__(
            n_var=self.n_var_s,
            n_obj=self.nn,
            n_constr=0,
            xl=self.l_s,
            xu=self.r_s,
            **kwargs,
        )
        return

    @staticmethod
    def plot_hist(hist_f, time=None, xaxis_time=False, marker="r--*", **kwargs):
        fig = plt.figure(kwargs)
        lns = []
        if xaxis_time:
            lns += plt.plot(time, hist_f, marker, kwargs)
            plt.xlabel("Execution Time")
        else:
            lns += plt.plot(np.arange(1, len(hist_f) + 1), hist_f, marker, kwargs)
            plt.xlabel("Generation")
        plt.ylabel("$F$")
        plt.grid()
        fig.tight_layout()
        return fig, lns


def write_yaml(gas, filename):
    t_writer = ct.YamlWriter()
    t_writer.add_solution(gas)
    t_writer.to_file(filename)
    return


def x2yaml(gas_orig, x, filename, skip=None):
    if type(skip) == str:
        skip = np.loadtxt(skip)
    if skip is None:
        fac = x + 1
        gas_modd = rxns_yaml_arr_list(gas_orig, fac)
        write_yaml(gas_modd, filename)
        return
    input_s = np.zeros(len(skip))
    input_s[~skip] = x
    fac = input_s + 1
    gas_modd = rxns_yaml_arr_list(gas_orig, fac)
    write_yaml(gas_modd, filename)
    return


def inputgas(gas):
    if type(gas).__name__ == "str":
        return ct.Solution(gas)
    else:
        return gas
