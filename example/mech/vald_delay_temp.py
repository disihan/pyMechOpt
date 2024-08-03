import matplotlib.pyplot as plt
import numpy as np
import cantera as ct

from pyMechOpt.sim import sim_0d_orig, sim_0d_delay
from pyMechOpt.vald import vald_mech

gas_orig = "./gri30.yaml"
gas_rdct = "./gri30-r45.yaml"
gas_optd = "./mech.opt.nsga3_orig_T.yaml"
# list_gas = [gas_orig, gas_rdct, gas_optd]
list_gas = [gas_orig, gas_rdct, gas_optd]

temp_ini = np.linspace(800, 1000, 6)
ratio = 1.0 * np.ones_like(temp_ini)
pres = 3.0 * np.ones_like(temp_ini)
delay_list = []

for t_gas in list_gas:
    t_delay = vald_mech.calc_delay(t_gas, temp_ini, ratio, pres)
    delay_list.append(t_delay)

err_rdct = np.abs(delay_list[1] - delay_list[0]) / delay_list[0]
err_optd = np.abs(delay_list[2] - delay_list[0]) / delay_list[0]
plt.plot(temp_ini, err_rdct, "-or")
plt.plot(temp_ini, err_optd, "-ob")


plt.savefig("vald-delay-temp.pdf")
plt.show()
