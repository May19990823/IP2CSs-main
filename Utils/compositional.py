from __future__ import annotations

from functools import reduce
from itertools import product
from math import gcd


def generate_composition(elements, ratios):
    unique_formulas = set()
    formulas = []
    formula_dict_list = []

    for ratio_comb in product(ratios, repeat=len(elements)):
        gcd_ratio = reduce(gcd, ratio_comb)
        formula = ""
        formula_dict = {}

        for element, ratio in zip(elements, ratio_comb):
            simplified_ratio = ratio // gcd_ratio
            if simplified_ratio > 0:
                formula += f"{element}{simplified_ratio}"
                formula_dict[element] = simplified_ratio

        if formula not in unique_formulas:
            unique_formulas.add(formula)
            formulas.append(formula)
            formula_dict_list.append(formula_dict)

    return formulas, formula_dict_list


def generate_stoichiometric_info(species_list, stoichiometric_range_list, specific_stoichiometric_list):
    has_range = len(stoichiometric_range_list) != 0
    has_specific = len(specific_stoichiometric_list) != 0

    if has_range and not has_specific:
        return generate_composition(elements=species_list, ratios=stoichiometric_range_list)

    if has_specific and not has_range:
        formulas = []
        formula_dict_list = []
        for specific_stoichiometric in specific_stoichiometric_list:
            formula = ""
            formula_dict = {}
            for specie, atom_ratio in zip(species_list, specific_stoichiometric):
                formula += f"{specie}{atom_ratio}"
                formula_dict[specie] = atom_ratio
            formulas.append(formula)
            formula_dict_list.append(formula_dict)
        return formulas, formula_dict_list

    raise ValueError("StoichiometricRatio and SpecificStoichiometric must have exactly one non-empty value")


def generate_species(full_formula_dict):
    species_list = []
    for specie, num in full_formula_dict.items():
        species_list.extend([specie] * int(num))
    return species_list
