import numpy as np
import cantera as ct
import os
import sys
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize

from pyMechOpt.sim import calc_all_orig, calc_all_rdct, spcs_name_idx, calc_y_int, calc_err_all_list
from pyMechOpt.mod_rxns import get_factor_dim, rxns_yaml_arr_list


class basic_problem(ElementwiseProblem):
    def __init__(self, gas_orig, gas_rdct, temp_ini=np.array([800]), ratio=np.array([1]), pres=np.array([101325]),
                 spcs_int=["H2"], spcs_peak=["OH"], **kwargs):
        # global num, gen, total_num
        if type(gas_orig).__name__ == "str":
            gas_orig = ct.Solution(gas_orig)
        if type(gas_rdct).__name__ == "str":
            gas_rdct = ct.Solution(gas_rdct)

        self.gas_rdct = gas_rdct
        if (temp_ini.shape == pres.shape and pres.shape == ratio.shape) == False:
            sys.stderr.write(
                "Error occurred: The dimensions of the initial condition vectors for pressure, temperature, and mixing ratio must be equal.\n")
            exit()
        self.ratio = ratio
        self.pres = pres
        self.temp_ini = temp_ini
        self.nn = len(temp_ini)

        # self.delay_orig = calc_delay(gas_orig, ratio=ratio, temp_ini=temp_ini, pres_ini=pres)

        self.idx_int_orig = spcs_name_idx(gas_orig, spcs_int)
        self.idx_int_rdct = spcs_name_idx(gas_rdct, spcs_int)
        self.idx_peak_orig = spcs_name_idx(gas_orig, spcs_peak)
        self.idx_peak_rdct = spcs_name_idx(gas_rdct, spcs_peak)
        # self.range_input = range_input

        self.n_var_o = get_factor_dim(gas_rdct)
        self.n_var_s = self.n_var_o
        self.skip = np.zeros(self.n_var_o, dtype=bool)
        # if calc_orig:
        self.t_list_orig, \
            self.temp_list_orig, \
            self.y_list_int_orig, \
            self.y_list_peak_orig = calc_all_orig(gas_orig, ratio, temp_ini, pres, self.idx_int_orig,
                                                  self.idx_peak_orig)
        self.y_int_orig = calc_y_int(self.t_list_orig, self.y_list_int_orig)

        # self.l = (-1) * np.ones(self.n_var_o) * range_input
        # self.r = np.ones(self.n_var_o) * range_input

        self.hist_f = []
        self.hist_x = []
        self.num = 0
        self.gen = 0
        self.pop_size = 0
        self.hist_dir = "./hist/"
        self.res_dir = "./res/"
        self.mech_res = "mech.opt."
        self.out_init()
        return

    def input2gas(self, input):
        input_s = np.zeros(self.n_var_o)
        input_s[~self.skip] = input
        fac = input_s + 1
        gas_modd = rxns_yaml_arr_list(self.gas_rdct, fac)
        return gas_modd

    def val_mo(self, input):
        gas_modd = self.input2gas(input)
        t_list_modd, temp_list_modd, y_list_int_modd, y_list_peak_modd = calc_all_rdct(gas_modd, self.ratio,
                                                                                       self.temp_ini, self.pres,
                                                                                       self.idx_int_rdct,
                                                                                       self.idx_peak_rdct,
                                                                                       self.t_list_orig)
        err_int, err_peak = calc_err_all_list(self.t_list_orig, self.temp_list_orig, self.y_list_int_orig,
                                              self.y_list_peak_orig,
                                              t_list_modd, temp_list_modd, y_list_int_modd, y_list_peak_modd,
                                              self.y_int_orig)
        err_int = np.array(err_int)
        err_peak = np.array(err_peak)
        err_int = np.sum(err_int, axis=1)
        err_peak = np.sum(err_peak, axis=1)
        err = err_int + err_peak
        return err

    def val_so(self, input):
        err = self.val_mo(input)
        err = np.linalg.norm(err)
        return err

    def calc_skip(self, skip_num=None):
        print("Calculating the number of variables that can be skipped.")
        input = np.zeros(self.n_var_o)
        job = self.job_skip(input)
        self.skip[job == 0] = True

        nvar_zero = np.sum(self.skip)
        order = np.argsort(np.abs(job))
        ign_len = int((len(order) - nvar_zero) / 2)
        if skip_num is None:
            self.skip[order[:ign_len + nvar_zero]] = True
        else:
            self.skip[order[:skip_num]] = True
        print("Done. total number: %d." % (np.sum(self.skip)))
        np.savetxt(self.res_dir + "skip.dat", self.skip, fmt="%d")
        self.n_var_s = self.n_var_o - np.sum(np.asarray(self.skip[:self.n_var_o], dtype=int))
        self.l_s = (-1) * np.ones(self.n_var_s) * self.range_input
        self.r_s = np.ones(self.n_var_s) * self.range_input
        return

    def load_skip(self):
        print("Loading skip.dat.")
        file = self.res_dir + "skip.dat"
        self.skip = np.loadtxt(file, dtype=bool)
        self.n_var_s = self.n_var_o - np.sum(np.asarray(self.skip[:self.n_var_o], dtype=int))
        self.l_s = (-1) * np.ones(self.n_var_s) * self.range_input
        self.r_s = np.ones(self.n_var_s) * self.range_input
        print("Done. total number: %d." % (np.sum(self.skip)))

    def job_skip(self, input, job_fac=1e-04):
        print("Calculating Jacobian matrix.")
        # input_s = np.zeros(self.n_var_o)
        # input_s[~self.skip] = input
        res = np.zeros(self.n_var_s)

        for k in range(self.n_var_s):
            t_fac_1 = np.zeros_like(input)
            t_fac_2 = np.zeros_like(input)
            t_fac_1[:] = input[:]
            t_fac_2[:] = input[:]
            t_fac_1[k] = t_fac_1[k] + job_fac
            t_fac_2[k] = t_fac_2[k] - job_fac
            t_j = self.val_so(t_fac_1) - self.val_so(t_fac_2)
            res[k] = t_j / job_fac
            # print("No.%d: %.6e" % (k, res[k]))
        return res

    # def val_skip(self, input):
    #     input_s = np.zeros(self.n_var_o)
    #     input_s[~self.skip] = input
    #     return self.val_mo(input_s)

    def out_init(self):
        if os.path.exists(self.hist_dir) == False:
            os.mkdir(self.hist_dir)
        if os.path.exists(self.res_dir) == False:
            os.mkdir(self.res_dir)
        return

    def soo_init(self, **kwargs):
        self.out_init()
        super().__init__(n_var=self.n_var_s,
                         n_obj=1,
                         n_constr=0,
                         xl=self.l_s,
                         xu=self.r_s,
                         **kwargs)
        return

    def moo_init(self, **kwargs):
        self.out_init()
        super().__init__(n_var=self.n_var_s,
                         n_obj=self.nn,
                         n_constr=0,
                         xl=self.l_s,
                         xu=self.r_s,
                         **kwargs)
        return


def write_yaml(gas, filename):
    t_writer = ct.YamlWriter()
    t_writer.add_solution(gas)
    t_writer.to_file(filename)
    return
