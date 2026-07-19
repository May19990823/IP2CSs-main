import os
import json
import numpy as np
from ip4ch.Lattice_and_Grids import generate_orthorhombic_lattices, set_cell_volume, generate_hexagonal_lattices, \
    generate_grid_size, generate_phase_name
from Utils.compositional import generate_species
from Utils.compositional import generate_composition
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# 定义生成化学计量比列表的函数
def entry(species_list, sto_range):
    stoichiometric_dict_list = generate_composition(elements=species_list, ratios=sto_range, is_dict=True)
    stoichiometric_list = []
    for atom_num_dict in stoichiometric_dict_list:
        compound_name = ''.join(f"{specie}{num}" for specie, num in atom_num_dict.items())
        stoichiometric_list.append(compound_name)
    return stoichiometric_list, stoichiometric_dict_list


# 定义生成不同 Z 倍数的化学式字典列表和原子总数列表
def generate_z_entry(sto_dict, Z_list):
    new_sto_list = []
    atom_num_list = []
    for z in Z_list:
        new_sto_dict = {specie: num * z for specie, num in sto_dict.items()}
        total_atoms = sum(new_sto_dict.values())
        new_sto_list.append(new_sto_dict)
        atom_num_list.append(total_atoms)
    return new_sto_list, atom_num_list


# 定义生成空间群列表的函数
def generate_space_group(lattice_type):
    space_groups = {
        # "Orthorhombic": list(range(16, 75)),  # 16 ≤ SG < 75 → 16-74,
        "Orthorhombic": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 43],
        "Tetragonal": [75, 76, 77, 78, 79, 80, 81, 82, 83, 85],
        "Cubic": [195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 206],
        "Hexagonal": [143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 156, 157, 158, 162, 163, 191, 194]
    }
    """
    space_groups = {
        "Orthorhombic": list(range(16, 75)),  # 16 ≤ SG < 75 → 16-74
        "Tetragonal": list(range(75, 143)),  # 75 ≤ SG < 143 → 75-142
        "Cubic": list(range(195, 231)),  # 195 ≤ SG < 231 → 195-230
        "Hexagonal": list(range(143, 195))  # 143 ≤ SG < 195 → 143-194
    }"""


    return space_groups.get(lattice_type, [])


# 定义生成 JSON 数据的函数
def generate_data(index_start, comp, sto_dict, atom_number, lattice, lattice_type, phase_name, grid, sg_list,
                  perturbation=None):
    data = []
    if perturbation:
        for sg in sg_list:
            tmp_lattice = lattice.copy()
            for i in range(3):
                tmp_lattice[i] += np.random.uniform(-perturbation, perturbation)
            data.append({
                "Index": index_start,
                "Chemical Stoichiometric": comp,
                "Full Formula Dictionary": sto_dict,
                "Atom Number": atom_number,
                "Full Formula": phase_name,
                "Lattice Type": lattice_type,
                # "Lattice": tmp_lattice,
                "Lattice": list(tmp_lattice),
                "Grid Size": grid,
                "Space Group": sg,
            })
            index_start += 1

    else:
        for sg in sg_list:
            data.append({
                "Index": index_start,
                "Chemical Stoichiometric": comp,
                "Full Formula Dictionary": sto_dict,
                "Atom Number": atom_number,
                "Full Formula": phase_name,
                "Lattice Type": lattice_type,
                "Lattice": list(lattice),
                "Grid Size": grid,
                "Space Group": sg,

            })
            index_start += 1

    return data, index_start


def generate_search_entries(EntryDir, name, multiplicities, compound_type, min_lattice_length, min_volume_factor,
                            max_volume_factor, scaling_step, entry_id, stoichiometric_list=None,
                            stoichiometric_dict_list=None):
    # all_entry_data = []
    for reduced_formula, reduced_formula_dict in zip(stoichiometric_list, stoichiometric_dict_list):
        entry_data = []
        sto_dictionary, atom_number_list = generate_z_entry(sto_dict=reduced_formula_dict, Z_list=multiplicities)
        for i in range(len(sto_dictionary)):
            # phase = stoichiometric_list[i]
            phase = generate_phase_name(atom_dict=sto_dictionary[i])
            v_min, v_max = set_cell_volume(atom_dict=sto_dictionary[i], min_volume_factor=min_volume_factor,
                                           max_volume_factor=max_volume_factor, compound_type=compound_type)
            print(f'{sto_dictionary[i]} V min: {v_min}')
            print(f'{sto_dictionary[i]} V max: {v_max}')

            # 生成类立方晶格
            cubic_lattices, cubic_types = generate_orthorhombic_lattices(cell_volume_min=v_min,
                                                                         cell_volume_max=v_max,
                                                                         scaling_step=scaling_step,
                                                                         min_lattice=min_lattice_length,
                                                                         compound_type=compound_type)
            for lattice, lattice_type in zip(cubic_lattices, cubic_types):
                grid = generate_grid_size(l=lattice, l_t=lattice_type, min_lattice_l=min_lattice_length)
                sg_list = generate_space_group(lattice_type)

                new_data, entry_id = generate_data(entry_id, reduced_formula, sto_dictionary[i], atom_number_list[i],
                                                   lattice, lattice_type, phase, grid, sg_list,
                                                   perturbation=False)

                # all_entry_data.extend(new_data)
                entry_data.extend(new_data)
                print(f"type of new_data: {type(new_data)}")

            # 生成六角晶格
            hexagonal_lattices, hexagonal_types = generate_hexagonal_lattices(cell_volume_min=v_min,
                                                                              cell_volume_max=v_max,
                                                                              scaling_step=scaling_step,
                                                                              min_lattice=min_lattice_length,
                                                                              compound_type=compound_type)
            for lattice, lattice_type in zip(hexagonal_lattices, hexagonal_types):
                grid = generate_grid_size(l=lattice, l_t=lattice_type, min_lattice_l=min_lattice_length)
                sg_list = generate_space_group(lattice_type)
                new_data, entry_id = generate_data(entry_id, reduced_formula, sto_dictionary[i], atom_number_list[i],
                                                   lattice, lattice_type, phase, grid, sg_list, perturbation=False)
                # all_entry_data.extend(new_data)
                entry_data.extend(new_data)
                print(f"type of new_data: {type(new_data)}")
        print(f"{reduced_formula} Total entries: {len(entry_data)}")
        output_file = os.path.join(EntryDir, name, f'{reduced_formula}.json')
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entry_data, f, indent=4, ensure_ascii=False)
        print(f"Entries of {reduced_formula} have been saved to {output_file}")

    # print(f"Total entries: {len(all_entry_data)}")


def read_json_entries(filepath):
    """
    Generator to read JSON entries one by one to save memory.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            yield entry


def generate_lattice_matrix(lattice, lattice_type):
    """
    Generate a lattice matrix from either legacy lengths or a 3x3 matrix.
    """
    lattice_array = np.asarray(lattice, dtype=float)
    if lattice_array.shape == (3, 3):
        if not np.all(np.isfinite(lattice_array)):
            raise ValueError("Lattice matrix contains non-finite values")
        return lattice_array
    if lattice_array.shape != (3,):
        raise ValueError(f"Unsupported lattice shape: {lattice_array.shape}")

    lattice = lattice_array.tolist()
    lattice_matrix = np.zeros((3, 3))
    if lattice_type in ["Orthorhombic", "Tetragonal", "Cubic"]:
        for i in range(3):
            lattice_matrix[i][i] = lattice[i]
    elif lattice_type in ["Hexagonal"]:
        a = lattice[0]
        c = lattice[2]
        lattice_matrix[0][0] = a
        lattice_matrix[1][0] = -0.5 * a
        lattice_matrix[1][1] = np.sqrt(3) / 2 * a
        lattice_matrix[2][2] = c
    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")
    return lattice_matrix


def generate_fractional_coordinates(num_atoms):
    """
    Generate fractional coordinates for the given number of atoms.
    """
    return np.random.rand(num_atoms, 3)


def process_entry(entry):
    """
    Process a single entry and return a pymatgen Structure object.
    """
    try:
        num_atoms = int(entry["Atom Number"])
        fractional_coords = generate_fractional_coordinates(num_atoms)
        lattice_matrix = generate_lattice_matrix(
            lattice=entry["Lattice"],
            lattice_type=entry["Lattice Type"]
        )
        species = generate_species(full_formula_dict=entry["Full Formula Dictionary"])
        structure = Structure(
            lattice=lattice_matrix,
            species=species,
            coords=fractional_coords
        )
        return structure
    except Exception as e:
        print(f"Error processing entry {entry}: {e}")
        return None
