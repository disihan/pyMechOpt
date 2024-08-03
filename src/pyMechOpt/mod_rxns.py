import cantera as ct
import numpy as np
import warnings


def get_factor_dim(t_gas):
    species = t_gas.species()
    reactions = t_gas.reactions()
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=species, reactions=reactions)
    rxns_orig = gas.reactions()
    p = 0
    for k in range(gas.n_reactions):
        t_rxn = rxns_orig[k]
        # type_rxns = gas.reaction_type(k)
        type_rxns = t_rxn.reaction_type
        if type_rxns == "Arrhenius":
            rate_a = t_rxn.rate.pre_exponential_factor
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent
            p = p + 1
            rate_e = t_rxn.rate.activation_energy
            p = p + 1
        elif type_rxns == "three-body-Arrhenius":
            rate_a = t_rxn.rate.pre_exponential_factor
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent
            p = p + 1
            rate_e = t_rxn.rate.activation_energy
            p = p + 1
        elif type_rxns == "falloff-Troe" or type_rxns == "falloff-Lindemann":
            lowrate_a = t_rxn.rate.low_rate.pre_exponential_factor
            p = p + 1
            lowrate_b = t_rxn.rate.low_rate.temperature_exponent
            p = p + 1
            lowrate_e = t_rxn.rate.low_rate.activation_energy
            p = p + 1
            highrate_a = t_rxn.rate.high_rate.pre_exponential_factor
            p = p + 1
            highrate_b = t_rxn.rate.high_rate.temperature_exponent
            p = p + 1
            highrate_e = t_rxn.rate.high_rate.activation_energy
            p = p + 1
        else:
            warnings.warn("Unsupported reaction type " + type_rxns + ".")
    return p


def rxns_yaml_arr_list(t_gas, factor):
    species = t_gas.species()
    reactions = t_gas.reactions()
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=species, reactions=reactions)
    rxns_orig = gas.reactions()
    rxns_modd = []
    p = 0
    for k in range(gas.n_reactions):
        t_rxn = rxns_orig[k]
        str_equ = t_rxn.equation
        # type_rxns = gas.reaction_type(k)
        type_rxns = t_rxn.reaction_type
        t_dup = t_rxn.duplicate
        str_dup = ""
        if t_dup:
            str_dup = ",\nduplicate: true"
        if type_rxns == "Arrhenius":
            rate_a = t_rxn.rate.pre_exponential_factor * factor[p]
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent * factor[p]
            p = p + 1
            rate_e = t_rxn.rate.activation_energy * factor[p]
            p = p + 1
            str_rate = '{A: ' + str(rate_a) + ', b: ' + str(rate_b) + ', Ea: ' + str(rate_e) + '}'
            str_cti = '{equation: ' + str_equ + ',\n' + \
                      'rate-constant: ' + str_rate + str_dup + '}'
        elif type_rxns == "three-body-Arrhenius":
            str_eff = str(t_rxn.efficiencies)
            str_eff = str_eff.replace('\'', '')
            rate_a = t_rxn.rate.pre_exponential_factor * factor[p]
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent * factor[p]
            p = p + 1
            rate_e = t_rxn.rate.activation_energy * factor[p]
            p = p + 1
            str_rate = '[' + str(rate_a) + ',' + str(rate_b) + ',' + str(rate_e) + ']'
            str_cti = '{equation: ' + str_equ + ',\n' + \
                      'type: three-body,\n' + \
                      'rate-constant: ' + str_rate + ',\n' + \
                      'efficiencies: ' + str_eff + str_dup + '}'
            # print(idx)
            # print(str_cti)
            # return ct.Reaction.fromCti(str_cti)
        elif type_rxns == "falloff-Troe" or type_rxns == "falloff-Lindemann":
            # str_type_falloff = t_rxn.falloff.falloff_type
            array_para_falloff = t_rxn.rate.falloff_coeffs
            str_eff = str(t_rxn.efficiencies)
            str_eff = str_eff.replace('\'', '')
            lowrate_a = t_rxn.rate.low_rate.pre_exponential_factor * factor[p]
            p = p + 1
            lowrate_b = t_rxn.rate.low_rate.temperature_exponent * factor[p]
            p = p + 1
            lowrate_e = t_rxn.rate.low_rate.activation_energy * factor[p]
            p = p + 1
            highrate_a = t_rxn.rate.high_rate.pre_exponential_factor * factor[p]
            p = p + 1
            highrate_b = t_rxn.rate.high_rate.temperature_exponent * factor[p]
            p = p + 1
            highrate_e = t_rxn.rate.high_rate.activation_energy * factor[p]
            p = p + 1
            str_lowrate = '{A: ' + str(lowrate_a) + ', b: ' + str(lowrate_b) + ', Ea: ' + str(lowrate_e) + '}'
            str_highrate = '{A: ' + str(highrate_a) + ', b: ' + str(highrate_b) + ', Ea: ' + str(highrate_e) + '}'
            str_cti = '{equation: ' + str_equ + ',\n' + \
                      'type: falloff,\n' + \
                      'low-P-rate-constant: ' + str_lowrate + ',\n' + \
                      'high-P-rate-constant: ' + str_highrate + ',\n'
            if type_rxns == "falloff-Troe":
                str_cti = str_cti + \
                          'Troe: {' + \
                          'A: ' + str(array_para_falloff[0]) + \
                          ', T3: ' + str(array_para_falloff[1]) + \
                          ', T1: ' + str(array_para_falloff[2]) + \
                          ', T2: ' + str(array_para_falloff[3]) + '},\n'
            str_cti = str_cti + 'efficiencies: ' + str_eff + str_dup + '}'
            # print(str_cti)
        tt_rxn = ct.Reaction.from_yaml(str_cti, gas)
        rxns_modd.append(tt_rxn)
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=species, reactions=rxns_modd)
    return gas


def rxns_yaml_arr(t_gas, factor_a, factor_b, factor_e):
    species = t_gas.species()
    reactions = t_gas.reactions()
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=species, reactions=reactions)
    rxns_orig = gas.reactions()
    rxns_modd = []
    for k in range(gas.n_reactions):
        # print('Modifying ' + str(k) + 'th reactions:')
        # print('Factor: ' + str(factor_a[k]))
        t_rxn = rxns_orig[k]
        str_equ = t_rxn.equation
        type_rxns = gas.reaction_type(k)
        if type_rxns == 1:
            rate_a = t_rxn.rate.pre_exponential_factor * factor_a[k]
            rate_b = t_rxn.rate.temperature_exponent * factor_b[k]
            rate_e = t_rxn.rate.activation_energy * factor_e[k]
            str_rate = '{A: ' + str(rate_a) + ', b: ' + str(rate_b) + ', Ea: ' + str(rate_e) + '}'
            str_cti = '{equation: ' + str_equ + ',\n' + \
                      'rate-constant: ' + str_rate + '}'
        if type_rxns == 2:
            str_eff = str(t_rxn.efficiencies)
            str_eff = str_eff.replace('\'', '')
            rate_a = t_rxn.rate.pre_exponential_factor * factor_a[k]
            rate_b = t_rxn.rate.temperature_exponent * factor_b[k]
            rate_e = t_rxn.rate.activation_energy * factor_e[k]
            str_rate = '[' + str(rate_a) + ',' + str(rate_b) + ',' + str(rate_e) + ']'
            str_cti = '{equation: ' + str_equ + ',\n' + \
                      'type: three-body,\n' + \
                      'rate-constant: ' + str_rate + ',\n' + \
                      'efficiencies: ' + str_eff + '}'
            # print(idx)
            # print(str_cti)
            # return ct.Reaction.fromCti(str_cti)
        if type_rxns == 4:
            str_type_falloff = t_rxn.falloff.falloff_type
            array_para_falloff = t_rxn.falloff.parameters
            str_eff = str(t_rxn.efficiencies)
            str_eff = str_eff.replace('\'', '')
            lowrate_a = t_rxn.low_rate.pre_exponential_factor * factor_a[k]
            lowrate_b = t_rxn.low_rate.temperature_exponent * factor_b[k]
            lowrate_e = t_rxn.low_rate.activation_energy * factor_e[k]
            highrate_a = t_rxn.high_rate.pre_exponential_factor * factor_a[k]
            highrate_b = t_rxn.high_rate.temperature_exponent * factor_b[k]
            highrate_e = t_rxn.high_rate.activation_energy * factor_e[k]
            str_lowrate = '{A: ' + str(lowrate_a) + ', b: ' + str(lowrate_b) + ', Ea: ' + str(lowrate_e) + '}'
            str_highrate = '{A: ' + str(highrate_a) + ', b: ' + str(highrate_b) + ', Ea: ' + str(highrate_e) + '}'
            str_cti = '{equation: ' + str_equ + ',\n' + \
                      'type: falloff,\n' + \
                      'low-P-rate-constant: ' + str_lowrate + ',\n' + \
                      'high-P-rate-constant: ' + str_highrate + ',\n'
            if str_type_falloff == 'Troe':
                str_cti = str_cti + \
                          'Troe: {' + \
                          'A: ' + str(array_para_falloff[0]) + \
                          ', T3:' + str(array_para_falloff[1]) + \
                          ', T1:' + str(array_para_falloff[2]) + \
                          ', T2:' + str(array_para_falloff[3]) + '}\n'
            str_cti = str_cti + 'efficiencies: ' + str_eff

        tt_rxn = ct.Reaction.fromYaml(str_cti, gas)
        rxns_modd.append(tt_rxn)
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=species, reactions=rxns_modd)
    return gas
