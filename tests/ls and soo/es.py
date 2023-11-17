import numpy as np

from pyMechOpt.single_obj import so_problem

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

mech_detailed = '../mech/gri30.yaml'  # detailed mech
mech_reduced = '../mech/gri30-r45.yaml'  # reduced mech

opt_prob = so_problem(mech_detailed, mech_reduced, temperature, ratio, pressure, spcs_int, spcs_peak)

try:
    opt_prob.load_skip()
except:
    opt_prob.calc_skip(skip_num=50)

from pymoo.algorithms.soo.nonconvex.es import ES
algorithm=ES(n_offsprings=200, rule=1.0 / 7.0)

opt_prob.run(algorithm=algorithm,
             max_gen=40)
