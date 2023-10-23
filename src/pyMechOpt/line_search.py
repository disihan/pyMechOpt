import numpy as np
import cantera as ct
import sys
import random
import os

from pyMechOpt.basic import basic_problem
from pyMechOpt.basic import write_yaml


class ls_problem(basic_problem):
    def __init__(self, gas_orig, gas_rdct, temp_ini, ratio, pres, spcs_int, spcs_peak,
                 max_step=0.05, eps=1e-08, **kwargs):
        self.max_step = max_step
        self.eps = eps
        self.algorithm_list = ["GD", "CG", "Mini-batch", "CD"]
        super().__init__(gas_orig, gas_rdct, temp_ini, ratio, pres, spcs_int, spcs_peak, range_input=0.1,
                         **kwargs)

    def run(self, algorithm="GD", max_gen=400, max_step=0.05, **kwargs):
        self.out_init()
        self.ls_init()
        if (algorithm in self.algorithm_list) == False:
            sys.stderr.write("Error occurred: Unsupported algorithm.\n")
            exit()
        self.iter = 0
        self.x = np.zeros(self.n_var_s)
        self.f = self.val_so(self.x)
        self.f_best = self.f
        self.x_best = np.zeros(self.n_var_s)
        self.job = None
        print("Init F=%.6e" % self.f)
        while self.iter < max_gen:
            if algorithm == "GD":
                self.gd_iter(max_step)
            elif algorithm == "CG":
                self.cg_iter(max_step)
            elif algorithm == "Mini-batch":
                self.mini_batch_iter(max_step, **kwargs)
            elif algorithm == "CD":
                self.cd_iter(max_step)

            if self.f_best < self.f:
                self.f_best = self.f
                self.x_best[:] = self.x[:]
            with open(self.hist_dir + "hist_x.txt", "a") as t_f:
                np.savetxt(t_f, self.x, newline=' ')
                t_f.write("\n")
            with open(self.hist_dir + "hist_f.txt", "a") as t_f:
                np.savetxt(t_f, np.array([self.f]))

            print(algorithm + " step: %d, F=%.6f" % (self.iter, self.f))
            self.iter += 1

        with open(self.res_dir + "res_x.txt", "a") as t_f:
            np.savetxt(t_f, self.x_best, newline=' ')
            t_f.write("\n")
        with open(self.res_dir + "res_f.txt", "a") as t_f:
            np.savetxt(t_f, np.array([self.f_best]))

        t_gas_modd = self.input2gas(self.x_best)
        write_yaml(t_gas_modd, self.res_dir + self.mech_res + ".yaml")
        return

    def out_init(self):
        super().out_init()
        if os.path.exists(self.hist_dir + "hist_x.txt") == True:
            os.remove(self.hist_dir + "hist_x.txt")
        if os.path.exists(self.hist_dir + "hist_f.txt") == True:
            os.remove(self.hist_dir + "hist_f.txt")
        return

    def ls_init(self):
        self.ls_k = (np.sqrt(5) - 1) / 2
        self.ls_k = np.array([0, 1 - self.ls_k, self.ls_k, 1])
        return

    def line_search_gss(self, x, direction, max_step, symm=False):
        print("Line search.")
        x_list = []
        direction = direction / np.max(np.abs(direction))
        if not symm:
            l = 0
        else:
            l = -max_step
        r = max_step
        x_step = max_step * self.ls_k
        for k in range(len(self.ls_k)):
            x_list.append(x + x_step[k] * direction)
        ls_iter = 0
        f_list = np.zeros(4)
        for k in range(len(x_list)):
            f_list[k] = (self.val_so(x_list[k]))
        while r - l > self.eps:

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
            # print(("Line search iter: %d: " + str(f_list)) % ls_iter)
        return x_b, f_b, step

    def gd_iter(self, max_step):
        self.job = self.job_skip(self.x)
        self.x, self.f, t_step = self.line_search_gss(self.x, -self.job, max_step)
        # self.max_step = t_step * 4
        return

    def cg_iter(self, max_step):
        g = self.job_skip(self.x)
        if self.job is None:
            p = g
        else:
            p = ((g - self.job).transpose() * g / (self.job.transpose() * self.job))[0, 0]
        self.x, self.f, t_step = self.line_search_gss(self.x, -p, max_step)
        return

    def job_batch(self, input, batch, job_fac=1e-04):
        print("Calculating Jacobian matrix.")
        res = np.zeros(self.n_var_s)

        for k in range(len(batch)):
            t_fac_1 = np.zeros_like(input)
            t_fac_2 = np.zeros_like(input)
            t_fac_1[:] = input[:]
            t_fac_2[:] = input[:]
            t_fac_1[batch[k]] = t_fac_1[batch[k]] + job_fac
            t_fac_2[batch[k]] = t_fac_2[batch[k]] - job_fac
            t_j = self.val_so(t_fac_1) - self.val_so(t_fac_2)
            res[batch[k]] = t_j / job_fac
            # print("No.%d: %.6e" % (k, res[k]))
        return res

    def mini_batch_iter(self, max_step, **kwargs):
        n_batch = kwargs.get("n_batch")
        t_list = np.arange(self.n_var_s, dtype=int)
        batch = random.sample(list(t_list), n_batch)
        self.job = self.job_batch(self.x, batch)
        self.x, self.f, t_step = self.line_search_gss(self.x, -self.job, max_step)
        return

    def cd_iter(self, max_step):
        self.job = np.zeros(self.n_var_s)
        t_n = self.iter % self.n_var_s
        self.job[t_n] = 1
        self.x, self.f, t_step = self.line_search_gss(self.x, self.job, max_step, symm=True)
        return
