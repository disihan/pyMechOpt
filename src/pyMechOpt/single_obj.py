import numpy as np
import cantera as ct
import sys
import os
import time

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pyMechOpt.basic import basic_problem
from pyMechOpt.basic import write_yaml


class so_problem(basic_problem):
    def __init__(self, gas_orig, gas_rdct, temp_ini=np.array([800]), ratio=np.array([1]), pres=np.array([101325]),
                 spcs_int=["H2"], spcs_peak=["OH"], range_input=0.2, *args, **kwargs):
        super().__init__(gas_orig, gas_rdct, temp_ini=temp_ini, ratio=ratio, pres=pres,
                         spcs_int=spcs_int, spcs_peak=spcs_peak, *args, **kwargs)
        self.range_input = range_input
        self.l_s = (-1) * np.ones(self.n_var_o) * range_input
        self.r_s = np.ones(self.n_var_o) * range_input
        return

    def _evaluate(self, x, out, *args, **kwargs):
        self.hist_x.append(x)
        f = np.array([self.val_so(x)])
        self.num += 1
        print(
            ("iter: %d,    num:%d,    f:" + np.array2string(f, floatmode="fixed", precision=4)) % (self.gen, self.num))
        self.hist_f.append(f)
        if len(self.hist_f) == self.pop_size:
            self.time_1 = time.process_time() - self.time_0
            self.gen = self.gen + 1
            np.savetxt(self.hist_dir + 'gen_' + str(self.gen) + '_x.dat', self.hist_x)
            np.savetxt(self.hist_dir + 'gen_' + str(self.gen) + '_f.dat', self.hist_f)
            t_f_best_idx = np.argmin(self.hist_f)
            t_f_best = self.hist_f[t_f_best_idx]
            t_x_best = self.hist_x[t_f_best_idx]
            with open(self.hist_dir + "hist_x.dat", "a") as t_f:
                np.savetxt(t_f, t_x_best, newline=' ')
                t_f.write("\n")
            with open(self.hist_dir + "hist_f.dat", "a") as t_f:
                np.savetxt(t_f, np.array([self.gen, t_f_best, self.time_1]), newline=' ')
                t_f.write("\n")
            self.hist_f = []
            self.hist_x = []
            self.num = 0
        out["F"] = f

    def out_init(self):
        super().out_init()
        if os.path.exists(self.hist_dir + "hist_x.dat") == True:
            os.remove(self.hist_dir + "hist_x.dat")
        if os.path.exists(self.hist_dir + "hist_f.dat") == True:
            os.remove(self.hist_dir + "hist_f.dat")
        return

    def run(self, algorithm="GA", seed=1, max_gen=400, pop_size=200, **kwargs):
        super().soo_init(**kwargs)
        self.time_0 = time.process_time()
        if type(algorithm).__name__ == "str":
            if algorithm == "GA":
                algorithm = GA(pop_size=pop_size)
            elif algorithm == "DE":
                algorithm = DE(pop_size=pop_size)
            elif algorithm == "PSO":
                algorithm = PSO(pop_size=pop_size)

        self.pop_size = algorithm.pop_size
        print("Pop size: %d" % (self.pop_size))
        print("START OPTIMIZATION.")
        res = minimize(self, algorithm, seed=seed, termination=('n_gen', max_gen))
        print("DONE OPTIMIZATION")
        np.savetxt(self.res_dir + "res_X.dat", res.X)
        np.savetxt(self.res_dir + "res_F.dat", res.F)
        t_X = res.X
        t_gas_modd = self.input2gas(t_X)
        write_yaml(t_gas_modd, self.res_dir + self.mech_res + "yaml")
        return res
