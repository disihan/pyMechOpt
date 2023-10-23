import numpy as np
import cantera as ct

from pyMechOpt.multi_obj import mo_problem

pressure = 1e05 * np.array([1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
temperature = np.array([900, 900, 900, 1000, 1000, 1000, 900, 900, 900, 1000, 1000, 1000])
ratio = np.array([0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4])
# pressure = 1e05 * np.array([100, 100, 100])
# temperature = np.array([1000, 1000, 1000])
# ratio = np.array([0.25, 1, 4])

spcs_int = ['CH4',
            'O2']
spcs_peak = ['CH2O',
             'CO',
             'CO2']

mech_detailed = 'mech/gri30.yaml'  # detailed mech
mech_reduced = 'mech/gri30-r45.yaml'  # reduced mech

opt_prob = mo_problem(mech_detailed, mech_reduced, temperature, ratio, pressure, spcs_int, spcs_peak)
# opt_prob.calc_skip(skip_num=50)
opt_prob.load_skip()
opt_prob.run(algorithm="MOEAD",
             max_gen=10,
             pop_size=200)
