import numpy as np
import cantera as ct

from pyMechOpt.single_obj import so_problem

pres = 1e05 * np.array([1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
temp_ini = np.array([900, 900, 900, 1000, 1000, 1000, 900, 900, 900, 1000, 1000, 1000])
ratio_eq = np.array([0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4])

spcs_int = ['CH4',
            'O2']
spcs_peak = ['CH2O',
             'CO',
             'CO2']

gas_orig = ct.Solution('mech/gri30.yaml')
gas_rdct = ct.Solution('mech/gri30-r45.yaml')

opt_prob = so_problem(gas_orig, gas_rdct, temp_ini, ratio_eq, pres, spcs_int, spcs_peak, range_input=0.2)
# opt_prob.calc_skip()
opt_prob.load_skip()
opt_prob.run(algorithm="PSO", max_gen=10, pop_size=200)
