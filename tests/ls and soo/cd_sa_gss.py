import numpy as np
import cantera as ct

from pyMechOpt.line_search import ls_problem

pres = 1e05 * np.array([1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
temp_ini = np.array([900, 900, 900, 1000, 1000, 1000, 900, 900, 900, 1000, 1000, 1000])
ratio_eq = np.array([0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4])
spcs_int = ['CH4',
            'O2']
spcs_peak = ['CH2O',
             'CO',
             'CO2']
6
gas_orig = ct.Solution('../mech/gri30.yaml')
gas_rdcd = ct.Solution('../mech/gri30-r45.yaml')

opt_prob = ls_problem(gas_orig, gas_rdcd, temp_ini, ratio_eq, pres, spcs_int, spcs_peak)

try:
    opt_prob.load_skip()
except:
    opt_prob.calc_skip(skip_num=50)

opt_prob.run(algorithm="CD", max_gen=300, max_step=0.1, ls_method="GSS",order="sensitivity")
