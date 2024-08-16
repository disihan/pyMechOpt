import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import cantera as ct
import sys
import os
import matplotlib.pyplot as plt

from pyMechOpt.basic import basic_problem
from pyMechOpt.basic import write_yaml, inputgas
from pyMechOpt.sim import sim_0d_delay

pic_type = ".pdf"
dir_save = "../../../figures/"
mech_dir = "../mech/"

config = {
    "text.usetex": True,
    # "font.family": "sans-serif",
    "font.family": "arial",
    # "font.sans-serif": "SimSun",
    # "font.size": 13,
    "mathtext.fontset": "stix",
}
# rcParams.update(config)
# rcParams["backend"] = "Qt5Agg"
# rcParams["lines.markersize"] = 5


# config = {
#     "text.usetex": True,
#     # "font.size": 13,
#     "mathtext.fontset": "stix",
# }
a = 4.5
b = 3.5

rcParams.update(config)


class vald_mech:
    def __init__(self, gas_orig, gas_optd, *args, **kwargs):
        if type(gas_orig).__name__ == "str":
            gas_orig = ct.Solution(gas_orig)
        if type(gas_optd).__name__ == "str":
            gas_optd = ct.Solution(gas_optd)
        self.gas_orig = gas_orig
        self.gas_optd = gas_optd
        return

    def err_delay_ratio_pres(
        self,
        temp_ini=800,
        ratio=np.array([1]),
        pres=np.array([101325]),
        delay_orig=None,
    ):
        delay_optd = np.zeros([len(pres), len(ratio)])
        error_optd = np.zeros([len(pres), len(ratio)])

        if delay_orig is None:
            delay_orig = np.zeros([len(pres), len(ratio)])
            for k in range(len(pres)):
                for p in range(len(ratio)):
                    self.gas_orig.TP = temp_ini, pres[k]
                    self.gas_orig.set_equivalence_ratio(ratio[p], "CH4:1", "O2:1")
                    delay_orig[k][p] = sim_0d_delay(self.gas_orig, tol=5e-06)

        for k in range(len(pres)):
            for p in range(len(ratio)):
                self.gas_orig.TP = temp_ini, pres[k]
                self.gas_orig.set_equivalence_ratio(ratio[p], "CH4:1", "O2:1")
                self.gas_optd.TP = temp_ini, pres[k]
                self.gas_optd.set_equivalence_ratio(ratio[p], "CH4:1", "O2:1")

                delay_optd[k][p] = sim_0d_delay(self.gas_optd, tol=5e-06)
                error_optd[k][p] = (
                    abs(delay_orig[k][p] - delay_optd[k][p]) / delay_orig[k][p]
                )
                print(
                    "Calc: temp=%.1lf    pres=%.3e    ratio=%.2lf    delay_o=%lf    delay_optd=%lf"
                    % (temp_ini, pres[k], ratio[p], delay_orig[k][p], delay_optd[k][p])
                )
        return error_optd, delay_orig, delay_optd

    @staticmethod
    def plot_err_delay(
        ratio=np.array([1.0]), pres=np.array([101325]), error_optd=np.array([1.0])
    ):
        color = ["b", "g", "r", "c", "m", "y", "k", "w"]
        fig = plt.figure()
        for k in range(len(pres)):
            str_label = str(pres[k] / 1e06) + "MPa" + " Simplified"
            plt.plot(
                ratio,
                error_optd[k][:],
                color[k] + "-s",
                label="{:>7}".format(str_label),
            )
        plt.xlabel(r"$\mathrm{Ratio}$")
        plt.ylabel(r"$\mathrm{Error}$")
        plt.tight_layout()
        return fig
