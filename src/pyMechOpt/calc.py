import cantera as ct
import numpy as np
import math


def spcs_name_idx(gas, spcs_ipt):
    spcs_idx = []
    for m in range(len(spcs_ipt)):
        for k in range(gas.n_species):
            if gas.species_name(k) == spcs_ipt[m]:
                spcs_idx.append(k)
    return spcs_idx


def calc_error(gas_1, gas_2, f_1, f_2, spcs_ipt):
    spcs_idx_1 = spcs_name_idx(gas_1, spcs_ipt)
    spcs_idx_2 = spcs_name_idx(gas_2, spcs_ipt)
    n_spcs = len(spcs_idx_2)
    n_vars = n_spcs + 2
    grid = f_1.grid
    n_points = len(grid)
    grid_size = (grid[2:n_points] - grid[0:n_points - 2]) / 2
    err = []
    for k in range(n_vars):
        if k < n_spcs:
            tmp_y_1 = f_1.Y[spcs_idx_1[k]][1:n_points - 1]
            tmp_y_2 = f_2.Y[spcs_idx_2[k]][1:n_points - 1]
        elif k == n_spcs + 1:
            tmp_y_1 = f_1.T[1:n_points - 1]
            tmp_y_2 = f_2.T[1:n_points - 1]
        else:
            tmp_y_1 = f_1.velocity[1:n_points - 1]
            tmp_y_2 = f_2.velocity[1:n_points - 1]
        tmp_err_y = (tmp_y_2 - tmp_y_1) ** 2
        tmp_mse = tmp_err_y * grid_size
        tmp_mse = tmp_mse.sum()
        err.append(np.sqrt(tmp_mse))

    return err


def calc_error_0d(temp_list, y_list, q, temp_orig, y_orig, spcs_idx):
    # global t_orig, y_orig, temp_orig, spcs_idx
    temp_error = (((temp_list - temp_orig[q]) * 1. / temp_orig[q]) ** 2).mean()
    err = []
    err.append(temp_error)
    # print(y_list.shape)
    # print(y_orig[q].shape)
    for k in range(len(spcs_idx)):
        # print(k)
        err.append((((y_list[:, k] - y_orig[q][:, k]) * 1. / y_orig[q][:, k].max()) ** 2).mean())
    return err
