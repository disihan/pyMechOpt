import cantera as ct
import numpy as np
import sys
from numpy import abs, linspace, floor, max, min

from scipy import integrate

from pyMechOpt.calc import calc_error_0d
from pyMechOpt.mod_rxns import get_factor_dim, rxns_yaml_arr_list


def spcs_name_idx(gas, spcs_ipt):
    spcs_idx = []
    for m in range(len(spcs_ipt)):
        for k in range(gas.n_species):
            if gas.species_name(k) == spcs_ipt[m]:
                spcs_idx.append(k)
    return spcs_idx


def sim_1d(
    sim_gas,
    fuel,
    oxydizer,
    temp_ini=300,
    pres_ini=ct.one_atm,
    alpha=1,
    width=0.03,
    grid=None,
):
    # gas = ct.Solution(file_mech)
    gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=sim_gas.species(),
        reactions=sim_gas.reactions(),
    )
    gas.TP = temp_ini, pres_ini
    gas.set_equivalence_ratio(
        1 / alpha,
        fuel,
        oxydizer,
    )
    if grid is None:
        f = ct.FreeFlame(gas, width=width)
        f.set_refine_criteria(ratio=8, slope=0.06, curve=0.12)
    else:
        f = ct.FreeFlame(gas, grid=grid)
    # f.show_solution()
    f.transport_model = "Mix"
    f.solve(loglevel=0, auto=True)
    return f


def sim_0d_orig(
    sim_gas,
    tol=1e-04,
):
    t_gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=sim_gas.species(),
        reactions=sim_gas.reactions(),
    )
    t_gas.Y = sim_gas.Y
    t_gas.TP = sim_gas.T, sim_gas.P
    r = ct.IdealGasConstPressureReactor(t_gas)
    sim = ct.ReactorNet([r])
    t_list = [0]
    temp_list = [sim_gas.T]
    d_temp_d_t = []
    d_t_list = []
    y_list = []
    y_list.append(r.Y)
    while 1:
        sim.step()
        t_list.append(sim.time)
        temp_list.append(r.T)
        d_temp = temp_list[-1] - temp_list[-2]
        d_t = t_list[-1] - t_list[-2]
        d_t_list.append(d_t)
        d_temp_d_t.append(abs(d_temp / d_t))
        y_list.append(r.Y)
        # print(d_temp_d_t)
        if d_temp_d_t[-1] < tol * np.max(d_temp_d_t):
            # print(t_list[-1])
            return t_list, temp_list, np.array(y_list)


def sim_0d_rdct(sim_gas, t_max):
    t_gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=sim_gas.species(),
        reactions=sim_gas.reactions(),
    )
    t_gas.Y = sim_gas.Y
    t_gas.TP = sim_gas.T, sim_gas.P
    r = ct.IdealGasConstPressureReactor(t_gas)
    sim = ct.ReactorNet([r])
    t_list = [0]
    temp_list = [sim_gas.T]
    y_list = []
    y_list.append(r.Y)
    while sim.time < t_max:
        sim.step()
        t_list.append(sim.time)
        temp_list.append(r.T)
        y_list.append(r.Y)
    return t_list, temp_list, np.array(y_list)


def sim_0d(sim_gas, t_orig):
    t_gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=sim_gas.species(),
        reactions=sim_gas.reactions(),
    )
    t_gas.Y = sim_gas.Y
    t_gas.TP = sim_gas.T, sim_gas.P
    r = ct.IdealGasConstPressureReactor(t_gas)
    sim = ct.ReactorNet([r])
    t_list = [0]
    temp_list = [sim_gas.T]
    d_temp_d_t = []
    d_t_list = []
    # y_list = np.array([])
    y_list = []
    y_list.append(r.Y)
    while 1:
        sim.step()
        t_list.append(sim.time)
        temp_list.append(r.T)
        d_temp = temp_list[-1] - temp_list[-2]
        d_t = t_list[-1] - t_list[-2]
        d_t_list.append(d_t)
        d_temp_d_t.append(abs(d_temp / d_t))
        y_list.append(r.Y)
        # print(d_temp_d_t)
        if t_list[-1] > t_orig[-1]:
            # print(t_list[-1])
            return t_list, temp_list, np.array(y_list)


def sim_0d_delay(
    sim_gas,
    delay="diff",
    tol=1e-04,
):
    t_gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=sim_gas.species(),
        reactions=sim_gas.reactions(),
    )
    t_gas.Y = sim_gas.Y
    t_gas.TP = sim_gas.T, sim_gas.P
    temp_ini = sim_gas.T
    r = ct.IdealGasConstPressureReactor(t_gas)
    sim = ct.ReactorNet([r])
    time = 0
    if delay == "400":
        while r.T < temp_ini + 400:
            t_time = time
            t_temp = r.T
            time = sim.step()
            temp = r.T
            # print(r.T)
        return np.interp(temp_ini + 400, [t_temp, temp], [t_time, time])
    elif delay == "diff":
        t_list = [0]
        temp_list = [sim_gas.T]
        d_temp_d_t = []
        d_t_list = []
        y_list = []
        y_list.append(r.Y)
        while 1:
            sim.step()
            t_list.append(sim.time)
            temp_list.append(r.T)
            d_temp = temp_list[-1] - temp_list[-2]
            d_t = t_list[-1] - t_list[-2]
            d_t_list.append(d_t)
            d_temp_d_t.append(abs(d_temp / d_t))
            y_list.append(r.Y)
            # print(d_temp_d_t)
            if d_temp_d_t[-1] < tol * np.max(d_temp_d_t):
                # print(t_list[-1])
                delay = calc_delay_diff(t_list, temp_list)
                return delay
    else:
        sys.stderr.write("Error occurred: unknown delay scheme.\n")
        exit()


def map_time_0d(t_list, temp_list, y_list, d_t, t_max):
    t_new = linspace(0, t_max, int(t_max / d_t))
    temp_new = np.interp(t_new, t_list, temp_list)
    y_new = np.zeros([len(t_new), y_list.shape[1]])
    for k in range(y_list.shape[1]):
        y_new[:, k] = np.interp(t_new, np.array(t_list), y_list[:, k])
    return t_new, temp_new, y_new


def calc_delay(
    t_gas, fuel, oxydizer, ratio=np.array([1]), temp_ini=1000, pres_ini=[ct.one_atm]
):
    gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=t_gas.species(),
        reactions=t_gas.reactions(),
    )
    n_ratio = len(ratio)
    n_pres = len(pres_ini)
    delay = []
    for k in range(n_pres):
        for p in range(n_ratio):
            gas.set_equivalence_ratio(
                ratio[p],
                fuel,
                oxydizer,
            )
            gas.TP = temp_ini, pres_ini[k]
            t_delay = sim_0d_delay(gas)
            delay.append(t_delay)
    return np.array(delay)


def calc_all_orig(
    t_gas, fuel, oxydizer, ratio, temp_ini, pres_ini, spcs_idx_int, spcs_idx_peak
):
    gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=t_gas.species(),
        reactions=t_gas.reactions(),
    )
    # n_ratio = len(ratio)
    n_pres = len(pres_ini)
    t_list_orig = []
    temp_list_orig = []
    y_list_int_orig = []
    y_list_peak_orig = []
    for k in range(n_pres):
        gas.set_equivalence_ratio(
            ratio[k],
            fuel,
            oxydizer,
        )
        gas.TP = temp_ini[k], pres_ini[k]
        t_list, temp_list, y_list = sim_0d_orig(gas)
        y_int_list = y_list[:, np.array(spcs_idx_int)]
        y_peak_list = y_list[:, np.array(spcs_idx_peak)]
        # dt = t_list[-1] / 5000
        # t_list_new, temp_list_new, y_list_new = map_time_0d(t_list, temp_list, y_list, dt, t_list[-1])
        t_list_orig.append(t_list)
        temp_list_orig.append(temp_list)
        y_list_int_orig.append(y_int_list)
        y_list_peak_orig.append(y_peak_list)
    return t_list_orig, temp_list_orig, y_list_int_orig, y_list_peak_orig


def calc_all_rdct(
    t_gas,
    fuel,
    oxydizer,
    ratio,
    temp_ini,
    pres_ini,
    spcs_idx_int,
    spcs_idx_peak,
    t_orig,
):
    gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=t_gas.species(),
        reactions=t_gas.reactions(),
    )
    # n_ratio = len(ratio)
    n_pres = len(pres_ini)
    error = []
    t_list_orig = []
    temp_list_orig = []
    y_list_int_orig = []
    y_list_peak_orig = []
    for k in range(n_pres):
        # for p in range(n_ratio):
        gas.set_equivalence_ratio(
            ratio[k],
            fuel,
            oxydizer,
        )
        gas.TP = temp_ini[k], pres_ini[k]
        t_list, temp_list, y_list = sim_0d_rdct(gas, t_orig[k][-1])
        y_int_list = y_list[:, np.array(spcs_idx_int)]
        y_peak_list = y_list[:, np.array(spcs_idx_peak)]
        # dt = t_list[-1] / 5000
        # t_list_new, temp_list_new, y_list_new = map_time_0d(t_list, temp_list, y_list, dt, t_list[-1])
        t_list_orig.append(t_list)
        temp_list_orig.append(temp_list)
        y_list_int_orig.append(y_int_list)
        y_list_peak_orig.append(y_peak_list)
    return t_list_orig, temp_list_orig, y_list_int_orig, y_list_peak_orig


def calc_y_int(t_list, y_list_int):
    res = []
    # print(t_list)
    # print(y_list_int)
    for k in range(len(t_list)):
        t_res = []
        for p in range(y_list_int[k].shape[1]):
            t_res.append(integrate.trapezoid(y_list_int[k][:, p], x=t_list[k]))
        res.append(t_res)
    return res


def calc_delay_diff(t, temp):
    t_arr = np.array(t)
    temp_arr = np.array(temp)
    d_temp = temp_arr[1:] - temp_arr[:-1]
    d_t = t_arr[1:] - t_arr[:-1]
    d_temp_d_t = d_temp / d_t
    delay_idx = np.argmax(np.abs(d_temp_d_t))
    delay = (t[delay_idx] + t[delay_idx + 1]) / 2
    return delay


def calc_temp_int(t_list, temp_list):
    res = []
    # print(t_list)
    # print(y_list_int)
    for k in range(len(t_list)):
        # print(np.sum(y_list_int[k][:, p] < 0))
        t_res = integrate.trapezoid(temp_list[k][:], x=t_list[k])
        res.append(t_res)
    return res


def calc_err_all(
    t_orig,
    temp_orig,
    y_int_orig,
    y_peak_orig,
    t_rdct,
    temp_rdct,
    y_int_rdct,
    y_peak_rdct,
    y_int_orig_res,
    temp_int_orig_res,
    norm=2.0,
    correct_peak=False,
    norm_int_int=True,
    calc=np.array([True, True, True, True]),
):
    nn = 4000
    t_max = t_orig[-1]
    t_new = linspace(0, t_max, nn + 1)
    dt = t_max / nn
    y_new_orig = np.zeros([len(t_new), y_int_orig.shape[1]])
    y_new_rdct = np.zeros([len(t_new), y_int_orig.shape[1]])
    err_int = []
    err_peak = []
    if calc[0]:
        for k in range(y_int_orig.shape[1]):
            y_new_orig[:, k] = np.interp(t_new, np.array(t_orig), y_int_orig[:, k])
            y_new_rdct[:, k] = np.interp(t_new, np.array(t_rdct), y_int_rdct[:, k])
            if norm_int_int:
                t_ = y_int_orig_res[k]
            else:
                t_ = np.max(y_int_orig[:, k]) * t_max
            err_int.append(
                (dt * np.sum(np.abs(y_new_rdct[:, k] - y_new_orig[:, k])) / (t_))
                ** norm
            )
    else:
        err_int = [0]
    if calc[1]:
        for k in range(y_peak_orig.shape[1]):
            idx_peak_orig = np.argmax(y_peak_orig[:, k])
            idx_peak_rdct = np.argmax(y_peak_rdct[:, k])
            if (
                np.abs(
                    (y_peak_orig[idx_peak_orig, k] - y_peak_orig[-1, k])
                    / y_peak_orig[idx_peak_orig, k]
                )
                < 1e-06
            ) and correct_peak:
                t_y_peak_orig = calc_delay_diff(t_orig, y_peak_orig[:, k])
                t_y_peak_rdct = calc_delay_diff(t_rdct, y_peak_rdct[:, k])
                err_peak.append(
                    (np.abs(t_y_peak_rdct - t_y_peak_orig) / t_y_peak_orig) ** norm
                )
            else:
                t_peak_orig = t_orig[idx_peak_orig]
                t_peak_rdct = t_rdct[idx_peak_rdct]
                # print("t_peak_orig=" + str(t_peak_orig) + "  t_peak_rdct=" + str(t_peak_rdct) + "  t_max=" + str(t_max))
                err_peak.append(
                    np.abs(((t_peak_rdct - t_peak_orig) / t_peak_orig) ** norm)
                )
    else:
        err_peak = [0]
    if calc[2]:
        temp_new_orig = np.interp(t_new, t_orig, temp_orig)
        temp_new_rdct = np.interp(t_new, t_rdct, temp_rdct)
        err_temp_int = (
            dt * np.sum(np.abs(temp_new_rdct[:] - temp_new_orig[:])) / temp_int_orig_res
        ) ** norm
    else:
        err_temp_int = 0
    if calc[3]:
        temp_peak_orig = calc_delay_diff(t_orig, temp_orig)
        temp_peak_rdct = calc_delay_diff(t_rdct, temp_rdct)
        err_temp_peak = (
            np.abs((temp_peak_orig - temp_peak_rdct) / temp_peak_orig) ** norm
        )
    else:
        err_temp_peak = 0
    return err_int, err_peak, err_temp_int, err_temp_peak

    # for k in range(y_int_orig.shape[1]):
    #     y_new_orig[:, k] = np.interp(t_new, np.array(t_orig), y_int_orig[:, k])
    #     y_new_rdct[:, k] = np.interp(t_new, np.array(t_rdct), y_int_rdct[:, k])
    #     err_int.append(
    #         (
    #             dt
    #             * np.sum(np.abs(y_new_rdct[:, k] - y_new_orig[:, k]))
    #             / y_int_orig_res[k]
    #         )
    #         ** 2
    #     )
    # for k in range(y_peak_orig.shape[1]):
    #     idx_peak_orig = np.argmax(y_peak_orig[:, k])
    #     idx_peak_rdct = np.argmax(y_peak_rdct[:, k])
    #     t_peak_orig = t_orig[idx_peak_orig]
    #     t_peak_rdct = t_rdct[idx_peak_rdct]
    #     # print("t_peak_orig=" + str(t_peak_orig) + "  t_peak_rdct=" + str(t_peak_rdct) + "  t_max=" + str(t_max))
    #     err_peak.append(((t_peak_rdct - t_peak_orig) / t_peak_orig) ** 2)
    # return err_int, err_peak, 0, 0


def calc_err_all_list(
    t_list_orig,
    temp_list_orig,
    y_list_int_orig,
    y_list_peak_orig,
    t_list_rdct,
    temp_list_rdct,
    y_list_int_rdct,
    y_list_peak_rdct,
    y_int_orig,
    temp_int_orig,
    **kwargs,
):
    norm = kwargs.get("norm", 2.0)
    correct_peak = kwargs.get("correct_peak", False)
    norm_int_int = kwargs.get("norm_int_int", True)
    calc = kwargs.get("weight", [1, 1, 1, 1])
    calc = np.array(calc).astype(bool)
    err_int = []
    err_peak = []
    err_temp_int = []
    err_temp_peak = []
    for k in range(len(t_list_orig)):
        t_err_int, t_err_peak, t_err_temp_int, t_err_temp_peak = calc_err_all(
            t_list_orig[k],
            temp_list_orig[k],
            y_list_int_orig[k],
            y_list_peak_orig[k],
            t_list_rdct[k],
            temp_list_rdct[k],
            y_list_int_rdct[k],
            y_list_peak_rdct[k],
            y_int_orig[k],
            temp_int_orig[k],
            norm=norm,
            correct_peak=correct_peak,
            norm_int_int=norm_int_int,
            calc=calc,
        )
        err_int.append(t_err_int)
        err_peak.append(t_err_peak)
        err_temp_int.append(t_err_temp_int)
        err_temp_peak.append(t_err_temp_peak)
    return err_int, err_peak, err_temp_int, err_temp_peak


def calc_all_error(
    t_gas,
    fuel,
    oxydizer,
    ratio,
    temp_ini,
    pres_ini,
    spcs_idx,
    t_orig,
    temp_orig,
    y_orig,
):
    gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=t_gas.species(),
        reactions=t_gas.reactions(),
    )
    n_ratio = len(ratio)
    n_pres = len(pres_ini)
    error = []
    q = 0
    for k in range(n_pres):
        for p in range(n_ratio):
            gas.set_equivalence_ratio(ratio[p], fuel, oxydizer)
            gas.TP = temp_ini, pres_ini[k]
            t_list, temp_list, y_list = sim_0d(gas, t_orig[q])
            y_list = y_list[:, spcs_idx]
            temp_list_new, y_list_new = map_modd_to_orig_time(
                t_list, temp_list, y_list, q, spcs_idx, t_orig
            )
            t_err = calc_error_0d(
                temp_list_new, y_list_new, q, temp_orig, y_orig, spcs_idx
            )
            error.extend(t_err)
            q = q + 1
    return error


def map_modd_to_orig_time(t_modd, temp_modd, y_modd, q, spcs_idx, t_orig):
    temp_new = np.interp(t_orig[q], t_modd, temp_modd)
    y_new = np.zeros([len(t_orig[q]), len(spcs_idx)])
    for k in range(len(spcs_idx)):
        y_new[:, k] = np.interp(t_orig[q], np.array(t_modd), y_modd[:, k])
    return temp_new, y_new


def calc_output_error_soo(delay_orig, delay_modd):
    return (delay_orig - delay_modd) * 1.0 / delay_orig
