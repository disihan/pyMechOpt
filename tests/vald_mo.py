import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pyMechOpt.multi_obj import mo_problem

pic_type = ".svg"
dir_save = './figures/'

config = {
    "font.family": 'serif',
    "font.size": 13,
    "mathtext.fontset": 'stix',
}

rcParams.update(config)
f_min = []
f_max = []
f_mean = []
f_err = []
hist_f = mo_problem.load_hist()

for k in range(len(hist_f)):
    t_f = hist_f[k]
    f_min.append(np.min(t_f, axis=0))
    f_max.append(np.max(t_f, axis=0))
    f_mean.append(np.mean(t_f, axis=0))
    f_err.append(np.std(t_f, axis=0))

f_min = np.array(f_min)
f_max = np.array(f_max)
f_mean = np.array(f_mean)
f_err = np.array(f_err)

n_obj = t_f.shape[1]
for k in range(n_obj):
    plt.errorbar(np.arange(1, len(hist_f) + 1), f_mean[:, k], capsize=7, label=r'$F_' + str(k + 1) + '$')

plt.xlabel("Generation")
plt.ylabel("$F$")
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig(dir_save + "MOEAD-12" + pic_type)

plt.figure(10, figsize=[10, 5])
plt.figure(20, figsize=[10, 5])
F = np.loadtxt("./res/res_F.dat")
F_imin_1 = np.expand_dims(np.min(F, axis=0), 0)
F_imax_1 = np.expand_dims(np.max(F, axis=0), 0)
F_imin = F_imin_1.repeat(F.shape[0], axis=0)
F_imax = F_imax_1.repeat(F.shape[0], axis=0)
F_o = (F - F_imin) / (F_imax - F_imin)
F_orig = np.loadtxt("./hist/f_orig.dat")
F_orig = np.expand_dims(F_orig, 0).repeat(F.shape[0], axis=0)
opt_ratio = F / F_orig
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
opt_ratio_mean = np.mean(opt_ratio, axis=0)
for k in range(opt_ratio.shape[0]):
    if (opt_ratio[k, :] < opt_ratio_mean).all():
        t_c = 'lightgrey'
    else:
        t_c = 'lightgrey'
    plt.figure(10)
    plt.plot(np.linspace(1, n_obj, n_obj), opt_ratio[k, :], 'x--', color=t_c, markeredgecolor=color[k % (len(color))])
    plt.figure(20)
    plt.plot(np.linspace(1, n_obj, n_obj), F_o[k, :], 'x--', color=t_c, markeredgecolor=color[k % (len(color))])
sct_x = []
for k in range(n_obj):
    sct_x.append("OBJ " + str(k + 1))
plt.figure(10)
plt.grid()
plt.xticks(np.linspace(1, n_obj, n_obj), labels=sct_x)
plt.yscale("log")
plt.ylabel(r"${F_{\rm{parato}}}/{F_0}$")
plt.tight_layout()
plt.savefig(dir_save + "MOEAD-ratio" + pic_type)

plt.figure(20)
plt.xticks(np.linspace(1, n_obj, n_obj), labels=sct_x)
plt.ylabel(r"$\tilde{F}$")
plt.tight_layout()
plt.savefig(dir_save + "MOEAD-F" + pic_type)
plt.show()