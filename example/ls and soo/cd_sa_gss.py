import numpy as np
import cantera as ct

from pyMechOpt.line_search import ls_problem

# pres = 1e05 * np.array([1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
# temp_ini = np.array([900, 900, 900, 1000, 1000, 1000, 900, 900, 900, 1000, 1000, 1000])
# ratio_eq = np.array([0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4, 0.25, 1, 4])
pressure = 1e05 * np.array([1, 1, 1, 100, 100, 100])
temperature = np.array([1000, 1000, 1000, 1000, 1000, 1000])
ratio = np.array([0.25, 1, 4, 0.25, 1, 4])

spcs_int = ["CH4", "O2"]
spcs_peak = ["CH2O", "CO", "CO2"]
gas_orig = ct.Solution("../mech/gri30.yaml")
gas_rdcd = ct.Solution("../mech/gri30-r45.yaml")
fuel = "CH4:1"
oxydizer = "O2:1"

opt_prob = ls_problem(
    gas_orig,
    gas_rdcd,
    fuel,
    oxydizer,
    temperature,
    ratio,
    pressure,
    spcs_int,
    spcs_peak,
    hist_dir="./hist_cd_sa_gss/",
    res_dir="./res_cd_sa_gss/",
)
try:
    opt_prob.load_skip()
except:
    opt_prob.calc_skip()

opt_prob.run(
    algorithm="CD", max_gen=2000, max_step=0.1, ls_method="GSS", order="sensitivity"
)
