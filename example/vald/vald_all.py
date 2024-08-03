import vald_init
from vald_init import pic_type, dir_save
import matplotlib.pyplot as plt
from pyMechOpt.multi_obj import mo_problem
from copy_so import copy_so
from vald_mo import (
    vald_mo,
    vald_delay,
    vald_1D,
    vald_0D,
    vald_delay_gas_alpha,
    vald_delay_single,
    vald_parallel,
    gen_best,
)

vald_mo("moead", "MOEAD",save_legend=True)
vald_mo("nsga3", "NSGA3")
vald_mo("ctaea", "CTAEA")

# gen_best("moead", "MOEAD")
# gen_best("ctaea", "CTAEA")
# gen_best("nsga3", "NSGA3")

# copy_so("cd")
# copy_so("ga")


# vald_delay("moead", "MOEAD")
# vald_delay("ctaea", "CTAEA")
# vald_delay("nsga3", "NSGA3")
# vald_delay_single(
#     [
#         "gri30-r45.yaml",
#         "mech.opt.cd.yaml",
#         "mech.opt.ga.yaml",
#         "mech.opt.nsga3.yaml",
#         "mech.opt.moead.yaml",
#         "mech.opt.ctaea.yaml",
#     ],
#     [
#         "GRI30-r45",
#         "GRI30-r45-CD",
#         "GRI30-r45-GA",
#         "GRI30-r45-NSGA3",
#         "GRI30-r45-MOEAD",
#         "GRI30-r45-CTAEA",
#     ],
# )

vald_parallel(
    [
        "gri30-r45.yaml",
        "mech.opt.cd.yaml",
        "mech.opt.ga.yaml",
        "mech.opt.nsga3.yaml",
        "mech.opt.moead.yaml",
        "mech.opt.ctaea.yaml",
    ],
    [
        "GRI30-r45",
        "GRI30-r45-CD",
        "GRI30-r45-GA",
        "GRI30-r45-NSGA3",
        "GRI30-r45-MOEAD",
        "GRI30-r45-CTAEA",
    ],
)

# vald_delay_gas_alpha(
#     [
#         "gri30.yaml",
#         "gri30-r45.yaml",
#         "mech.opt.nsga3.yaml",
#         "mech.opt.moead.yaml",
#         "mech.opt.ctaea.yaml",
#     ],
#     [
#         "GRI-Mech 3.0",
#         "GRI30-r45",
#         "GRI30-r45-NSGA3",
#         "GRI30-r45-MOEAD",
#         "GRI30-r45-CTAEA",
#     ],
# )
