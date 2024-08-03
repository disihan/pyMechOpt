import matplotlib.pyplot as plt
import numpy as np
import cantera as ct

from pyMechOpt.sim import sim_0d_orig, sim_0d_delay

gas_orig = "./gri30.yaml"
gas_rdct = "./gri30-r45.yaml"
gas_optd = "./mech.opt.nsga3_orig.yaml"
# list_gas = [gas_orig, gas_rdct, gas_optd]
list_gas = [gas_orig, gas_rdct, gas_optd]

temp_ini = 1000
ratio = 3.0
pres_init = 5e05

for t_gas in list_gas:
    t_gas = ct.Solution(t_gas)
    t_gas.TP = temp_ini, pres_init
    t_gas.set_equivalence_ratio(ratio, "CH4:1", "O2:1")
    t, temp, y = sim_0d_orig(t_gas)
    t = np.array(t)
    temp = np.array(temp)
    print(temp.shape)
    print(sim_0d_delay(t_gas))
    # plt.plot((t[1:] + t[:-1]) / 2.0, (temp[1:] - temp[:-1]) / (t[1:] - t[:-1]), "-o")
    plt.plot(t, temp, "-o")

plt.savefig("temp.pdf")
plt.show()
