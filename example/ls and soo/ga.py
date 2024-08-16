import numpy as np

from pyMechOpt.single_obj import so_problem

# pres = 1e05 * np.array([1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
# temp_ini = np.array([900, 900, 900, 1000, 1000, 1000, 900, 900, 900, 1000, 1000, 1000])
# ratio_eq = np.array([0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4])
pressure = 1e05 * np.array([1, 1, 1, 100, 100, 100])
temperature = np.array([1000, 1000, 1000, 1000, 1000, 1000])
ratio = np.array([0.25, 1, 4, 0.25, 1, 4])

spcs_int = ["CH4", "O2"]
spcs_peak = ["CH2O", "CO", "CO2"]

mech_detailed = "../mech/gri30.yaml"  # detailed mech
mech_reduced = "../mech/gri30-r45.yaml"  # reduced mech
fuel = "CH4:1"
oxydizer = "O2:1"

opt_prob = so_problem(
    mech_detailed,
    mech_reduced,
    fuel,
    oxydizer,
    temperature,
    ratio,
    pressure,
    spcs_int,
    spcs_peak,
    hist_dir="./hist_ga/",
    res_dir="./res_ga/",
)

try:
    opt_prob.load_skip()
except:
    opt_prob.calc_skip(skip_num=50)

opt_prob.run(algorithm="GA", max_gen=400, pop_size=200)
