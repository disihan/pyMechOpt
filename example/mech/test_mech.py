import cantera as ct
from pyMechOpt.sim import sim_0d_orig, spcs_name_idx
import matplotlib.pyplot as plt

gas_orig = ct.Solution("./gri30.yaml")
# gas_rdct = ct.Solution('mech/gri-drgep23-sa1459-pca0.5-r45.cti')
gas_rdct = ct.Solution("./gri30-r45.yaml")
name = ["GRI30", "GRI30-r45"]
spcs_name = ["CH4", "O2", "H2O", "CO", "CO2"]
pres = 1e05
temp_ini = 1000
ratio_eq = 1
mk = ["-", "--"]
k = 0
color = ["b", "g", "r", "c", "m", "y", "k", "w"]
for t_gas in [gas_orig, gas_rdct]:
    # plt.figure()
    t_idx = spcs_name_idx(t_gas, spcs_name)
    t_gas.TP = temp_ini, pres
    t_gas.set_equivalence_ratio(ratio_eq, "CH4:1", "O2:1")
    t_list, temp_list, y_list = sim_0d_orig(t_gas)
    for p in range(len(t_idx)):
        plt.plot(
            t_list,
            y_list[:, t_idx[p]],
            mk[k] + color[p],
            markersize=2,
            linewidth=2.5,
            label=name[k] + ":" + spcs_name[p],
        )
    k += 1

plt.xlim([0.2, 0.245])
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Y")
plt.tight_layout()
plt.savefig("test-0D-r45.svg")
plt.show()
