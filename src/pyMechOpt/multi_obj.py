import numpy as np
import cantera as ct
import sys
import os

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
        f = self.val_mo(x)
        self.num += 1
        print(
            ("iter: %d,    num:%d,    f:" + np.array2string(f, floatmode="fixed", precision=4)) % (self.gen, self.num))
        self.hist_f.append(f)
        if len(self.hist_f) == self.pop_size:
            self.gen = self.gen + 1
            np.savetxt(self.hist_dir + 'gen_' + str(self.gen) + '_x.dat', self.hist_x)
            np.savetxt(self.hist_dir + 'gen_' + str(self.gen) + '_f.dat', self.hist_f)
            self.hist_f = []
            self.hist_x = []
            self.num = 0
        out["F"] = f

    def run(self, algorithm="NSGA3", max_gen=400, pop_size=200, **kwargs):
        super().moo_init(**kwargs)
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
    def load_hist(load_x=False):
        k = 1
        hist_f = []
        while True:
            t_fname = 'gen_' + str(k) + '_f.dat'
            t_fname = "./hist/" + t_fname
            if os.path.exists(t_fname) == True:
                print("Loading history data: " + t_fname)
                t_f = np.loadtxt(t_fname)
                hist_f.append(t_f)
            else:
                return hist_f
            k = k + 1
