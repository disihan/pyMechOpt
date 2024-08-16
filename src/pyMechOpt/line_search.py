import time

import numpy as np
import cantera as ct
import sys
import random
import os
from scipy.interpolate import interp1d

from pyMechOpt.basic import basic_problem
from pyMechOpt.basic import write_yaml


class ls_problem(basic_problem):
    """
    A class for chemcal reaction mechanisms based on line search algorithm.
    The available algorithms are GD, CG, CD and mini-batch SGD.
    """

    def __init__(
        self,
        gas_orig,
        gas_rdct,
        fuel,
        oxydizer,
        temp_ini,
        ratio,
        pres,
        spcs_int,
        spcs_peak,
        max_step=0.05,
        eps=1e-08,
        step_ratio=10,
        **kwargs,
    ):
        self.max_step = max_step
        self.range_input = max_step * 4
        self.eps = eps
        self.step_ratio = step_ratio
        super().__init__(
            gas_orig,
            gas_rdct,
            fuel,
            oxydizer,
            temp_ini,
            ratio,
            pres,
            spcs_int,
            spcs_peak,
            **kwargs,
        )

    def run(self, algorithm="GD", max_gen=400, max_step=0.05, **kwargs):
        self.out_init()
        self.ls_init()
        self.time_0 = time.process_time()
        # if (algorithm in self.algorithm_list) == False:
        #     sys.stderr.write("Error occurred: Unsupported algorithm.\n")
        #     exit()
        if algorithm == "CD" or algorithm == "SCD":
            self.step_cd = np.ones(self.n_var_s) * self.max_step / 2
        else:
            self.step = self.max_step
        if algorithm == "CD":
            t_order = kwargs.get("order", "sequential")
            if t_order == "sequential":
                self.order = np.arange(0, self.n_var_s)
            elif t_order == "sensitivity":
                try:
                    self.order = np.loadtxt(self.res_dir + "order_skip.dat", dtype=int)
                except FileNotFoundError:
                    try:
                        print("Load order.txt.")
                        skip_num = np.sum(self.skip)
                        t_order = np.loadtxt(self.res_dir + "order.dat", dtype=int)
                        t_order = t_order[skip_num:]
                        t_var = np.sort(t_order)
                        # t_idx =
                        self.order = t_var.searchsorted(t_order)
                    except FileNotFoundError:
                        print("Sensitivity analysis.")
                        grad = self.grad(np.zeros(self.n_var_s))
                        self.order = np.argsort(np.abs(grad))
                        np.savetxt(self.res_dir + "sa_skip.dat", grad)
                    np.savetxt(self.res_dir + "order_skip.dat", self.order, fmt="%d")
                    self.order = self.order[::-1]

            else:
                sys.stderr.write("Unknown order scheme.")
                exit()

        self.iter = 0
        self.x = np.zeros(self.n_var_s)
        self.f = self.save_f_orig_so()
        self.f_best = self.f
        self.x_best = np.zeros(self.n_var_s)
        self.direction = None
        print("Init F=%.6e" % self.f)
        while self.iter < max_gen:
            if algorithm == "GD":
                self.gd_iter(max_step, **kwargs)
            elif algorithm == "CG":
                self.cg_iter(max_step, **kwargs)
            elif algorithm == "Mini-batch":
                self.mini_batch_iter(max_step, **kwargs)
            elif algorithm == "CD":
                self.cd_iter(max_step, **kwargs)
            elif algorithm == "SCD":
                self.scd_iter(max_step, **kwargs)
            else:
                sys.stderr.write("Error occurred: Unsupported algorithm.\n")
                exit()

            if self.f_best > self.f:
                self.f_best = self.f
                self.x_best[:] = self.x[:]
            self.time_1 = time.process_time() - self.time_0
            self.iter += 1
            with open(self.hist_dir + "hist_x.dat", "a") as t_f:
                np.savetxt(t_f, self.x, newline=" ")
                t_f.write("\n")
            with open(self.hist_dir + "hist.dat", "a") as t_f:
                np.savetxt(t_f, np.array([self.f, self.time_1]), newline=" ")
                t_f.write("\n")

            print(
                algorithm
                + " step: %d, F=%.6f, time=%f" % (self.iter, self.f, self.time_1)
            )

        with open(self.res_dir + "res_x.dat", "a") as t_f:
            np.savetxt(t_f, self.x_best, newline=" ")
            t_f.write("\n")
        with open(self.res_dir + "res_f.dat", "a") as t_f:
            np.savetxt(t_f, np.array([self.f_best]))

        t_gas_modd = self.input2gas(self.x_best)
        write_yaml(t_gas_modd, self.res_dir + self.mech_res + "yaml")
        return

    def out_init(self):
        super().out_init()
        if os.path.exists(self.hist_dir + "hist_x.dat") == True:
            os.remove(self.hist_dir + "hist_x.dat")
        if os.path.exists(self.hist_dir + "hist.dat") == True:
            os.remove(self.hist_dir + "hist.dat")
        return

    def ls_init(self):
        self.ls_k = (np.sqrt(5) - 1) / 2
        self.ls_k = np.array([0, 1 - self.ls_k, self.ls_k, 1])
        return

    def load_skip(self):
        print("Loading skip.dat.")
        file = self.res_dir + "skip.dat"
        self.skip = np.loadtxt(file, dtype=bool)
        self.n_var_s = self.n_var_o - np.sum(
            np.asarray(self.skip[: self.n_var_o], dtype=int)
        )
        print("Done. total number: %d." % (np.sum(self.skip)))
        return

    def line_search_gss(self, x, direction, max_step, symm, max_ls_step):
        print("Line search: GSS.")
        x_list = []
        direction = direction / np.max(np.abs(direction))
        if not symm:
            l = 0
        else:
            l = -max_step
        r = max_step
        x_step = (r - l) * self.ls_k + l
        for k in range(len(self.ls_k)):
            x_list.append(x + x_step[k] * direction)
        ls_iter = 0
        f_list = np.zeros(4)

        for k in range(len(x_list)):
            f_list[k] = self.val_so(x_list[k])
        if r - l <= self.eps:
            return x, self.val_so(x), 0
        while r - l > self.eps and ls_iter < max_ls_step:
            ls_iter += 1
            if f_list[2] > f_list[1]:
                l = x_step[0]
                r = x_step[2]
                x_step = (r - l) * self.ls_k + l
                x_list[1] = x + x_step[1] * direction
                x_list[2:] = x_list[1:3]
                f_list[1] = self.val_so(x_list[1])
                f_list[2:] = f_list[1:3]
            else:
                l = x_step[1]
                r = x_step[3]
                x_step = (r - l) * self.ls_k + l
                x_list[2] = x + x_step[2] * direction
                x_list[0:2] = x_list[1:3]
                f_list[0:2] = f_list[1:3]
                f_list[2] = self.val_so(x_list[2])
            t_idx = np.argmin(f_list)
            f_b = f_list[t_idx]
            x_b = x_list[t_idx]
            step = x_step[t_idx]
            print(
                "\titer: %d    x:%.6e    f:%.6e    idx:%d" % (ls_iter, step, f_b, t_idx)
            )
        return x_b, f_b, step

    def line_search_spline(
        self, x, direction, max_step, symm, n_init, n_p, max_ls_step
    ):
        print("Line search: spline.")
        x_list = []
        direction = direction / np.max(np.abs(direction))
        if not symm:
            l = 0
        else:
            l = -max_step
        r = max_step
        x_step = list(np.linspace(l, r, n_init))
        x_step_r = list(np.linspace(l, r, n_p))
        for k in range(len(x_step)):
            x_list.append(x + x_step[k] * direction)
        ls_iter = 0
        f_list = []
        if np.abs(r - l) < self.eps:
            return x, self.val_so(x), 0
        for k in range(len(x_list)):
            f_list.append(self.val_so(x_list[k]))
        while ls_iter < max_ls_step:
            ls_iter += 1
            t_func = interp1d(x_step, f_list, kind="cubic")
            f_r = t_func(x_step_r)
            t_f_min_idx = np.argmin(f_r)
            t_f_min = f_r[t_f_min_idx]
            t_f_max = np.max(f_r)
            t_x_step_min = x_step_r[t_f_min_idx]
            t_x_min = x + t_x_step_min * direction
            if (t_x_step_min in x_step) or (
                (t_f_max - t_f_min) / (np.abs(t_f_max)) < self.eps
            ):
                return t_x_min, t_f_min, t_x_step_min
            x_step.append(t_x_step_min)
            f_list.append(self.val_so(t_x_min))
            print(
                "\titer: %d    x:%.6e    f:%.6e    idx:%d"
                % (ls_iter, t_x_step_min, f_list[-1], t_f_min_idx)
            )
            if np.abs(x_step[-1] - x_step[-2]) < self.eps:
                return t_x_min, f_list[-1], x_step[-1]
        return t_x_min, t_f_min, t_x_step_min

    def linesearch(self, x, direction, max_step, symm=False, **kwargs):
        ls_method = kwargs.get("ls_method", "GSS")
        max_lsstep = kwargs.get("max_ls_step", 100)
        if ls_method == "GSS":
            return self.line_search_gss(x, direction, max_step, symm, max_lsstep)
        elif ls_method == "spline":
            n_init = kwargs.get("n_init", 5)
            n_p = kwargs.get("n_p", 100001)
            return self.line_search_spline(
                x, direction, max_step, symm, n_init, n_p, max_lsstep
            )
        else:
            sys.stderr.write("Unsupported line search method: " + str(ls_method))
            exit()

    def gd_iter(self, max_step, **kwargs):
        self.direction = self.grad(self.x, log=True)
        self.x, self.f, self.step = self.linesearch(
            self.x,
            -self.direction,
            min(self.step * self.step_ratio, self.max_step),
            **kwargs,
        )
        return

    def cg_iter(self, max_step, **kwargs):
        g = self.grad(self.x)
        if self.direction is None:
            p = g
        else:
            p = (
                (g - self.direction).transpose()
                * g
                / (self.direction.transpose() * self.direction)
            )[0, 0]
        self.x, self.f, t_step = self.linesearch(
            self.x, -p, min(self.step * self.step_ratio, self.max_step), **kwargs
        )
        return

    def grad_batch(self, input, batch, grad_fac=1e-04):
        print("Calculating Jacobian matrix.")
        res = np.zeros(self.n_var_s)

        for k in range(len(batch)):
            t_fac_1 = np.zeros_like(input)
            t_fac_2 = np.zeros_like(input)
            t_fac_1[:] = input[:]
            t_fac_2[:] = input[:]
            t_fac_1[batch[k]] = t_fac_1[batch[k]] + grad_fac
            t_fac_2[batch[k]] = t_fac_2[batch[k]] - grad_fac
            t_j = self.val_so(t_fac_1) - self.val_so(t_fac_2)
            res[batch[k]] = t_j / grad_fac
        return res

    def mini_batch_iter(self, max_step, **kwargs):
        n_batch = kwargs.get("n_batch")
        self.batch = self.random_batch(self.n_var_s, n_batch)
        self.direction = self.grad_batch(self.x, self.batch)
        self.x, self.f, t_step = self.linesearch(
            self.x,
            -self.direction,
            min(self.step * self.step_ratio, self.max_step),
            **kwargs,
        )
        return

    @staticmethod
    def random_batch(n_var_s, n_batch):
        t_list = np.arange(n_var_s, dtype=int)
        batch = random.sample(list(t_list), n_batch)
        return batch

    def cd_iter(self, max_step, **kwargs):
        self.direction = np.zeros(self.n_var_s)
        t_n = self.iter % self.n_var_s
        self.direction[self.order[t_n]] = 1
        self.x, self.f, self.step_cd[t_n] = self.linesearch(
            self.x,
            self.direction,
            min(self.step_cd[t_n] * self.step_ratio, max_step),
            symm=True,
            **kwargs,
        )
        self.step_cd[t_n] = np.abs(self.step_cd[t_n])
        return

    def scd_iter(self, max_step, **kwargs):
        self.direction = np.zeros(self.n_var_s)
        t_n = self.iter % self.n_var_s
        if t_n == 0:
            self.batch = self.random_batch(self.n_var_s, self.n_var_s)
        self.direction[self.batch[t_n]] = 1
        self.x, self.f, self.step_cd[t_n] = self.linesearch(
            self.x,
            self.direction,
            min(self.step_cd[t_n] * self.step_ratio, max_step),
            symm=True,
            **kwargs,
        )
        self.step_cd[t_n] = np.abs(self.step_cd[t_n])

    @staticmethod
    def load_hist(hist_dir="./hist/"):
        f_all = np.loadtxt(hist_dir + "hist.dat")
        f = f_all[:, 0]
        time = f_all[:, 1]
        f = list(f)
        time = list(time)
        f_orig = np.loadtxt(hist_dir + "f_orig.dat")
        f.insert(0, float(f_orig))
        time.insert(0, 0.0)
        return f, time
