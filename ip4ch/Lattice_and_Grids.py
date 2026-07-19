from pyxtal.database.element import Element
import math
import numpy as np
from collections import defaultdict


def lattices_reduce(ori_lattices):
    result_lattices = []
    a_b_dictionary = defaultdict(list)
    for lattice in ori_lattices:
        a_b_key = f'{lattice[0]}-{lattice[1]}'
        a_b_dictionary[a_b_key].append(lattice)
    print(a_b_dictionary)
    for key, lattice_group in a_b_dictionary.items():
        # 若当前a-b键下只有一个lattice，则直接保存
        if len(lattice_group) == 1:
            result_lattices.extend(lattice_group)
        else:
            if np.mod(len(lattice_group), 2) != 0:
                print(f"{key}: {len(lattice_group)}")
                for i in range(0, len(lattice_group), 2):
                    try:
                        if len(set(lattice_group[i])) == 1:
                            result_lattices.append(lattice_group[i])
                        elif len(set(lattice_group[i + 1])) == 1:
                            result_lattices.append(lattice_group[i] + 1)
                        tmp_lattice = []
                        ave_c = (lattice_group[i][2] + lattice_group[i + 1][2]) / 2
                        print(ave_c)
                        a_b = key.split("-")
                        for i in range(len(a_b)):
                            tmp_lattice.append(float(a_b[i]))
                        print(tmp_lattice)
                        tmp_lattice.append(ave_c)
                        print(tmp_lattice)
                        result_lattices.append(tmp_lattice)

                    except:
                        result_lattices.append(lattice_group[i])
            elif np.mod(len(lattice_group), 2) == 0:
                for i in range(0, len(lattice_group), 2):
                    if len(set(lattice_group[i])) == 1:
                        result_lattices.append(lattice_group[i])
                    elif len(set(lattice_group[i + 1])) == 1:
                        result_lattices.append(lattice_group[i+1])
                    tmp_lattice = []
                    ave_c = (lattice_group[i][2] + lattice_group[i + 1][2]) / 2
                    print(ave_c)
                    a_b = key.split("-")
                    for i in range(len(a_b)):
                        tmp_lattice.append(float(a_b[i]))
                    print(tmp_lattice)
                    tmp_lattice.append(ave_c)
                    print(tmp_lattice)
                    result_lattices.append(tmp_lattice)

    return result_lattices


def set_cell_volume(atom_dict, min_volume_factor, max_volume_factor, compound_type="alloy"):
    atomic_volume = 0

    for specie, num in atom_dict.items():
        if compound_type == "alloy":
            if Element(str(specie)).metallic_radius:
                atomic_volume += (4 / 3 * math.pi * Element(str(specie)).metallic_radius ** 3) * num
            else:
                atomic_volume += (4 / 3 * math.pi * Element(str(specie)).covalent_radius ** 3) * num
                print(f"{specie} is not a metallic element, using its covalent radius")

        elif compound_type == "semiconductor":
            atomic_volume += (4 / 3 * math.pi * Element(str(specie)).covalent_radius ** 3) * num

        else:
            raise ValueError(f"No {compound_type} Type")
    min_cell_volume = (min_volume_factor * atomic_volume) - 10
    max_cell_volume = (max_volume_factor * atomic_volume) + 10

    print(f'The volume of atoms is {atomic_volume}')
    print(f'The minimal volume of cell is {min_cell_volume}')
    print(f'The maximal volume of cell is {max_cell_volume}')

    return min_cell_volume, max_cell_volume


def generate_grid_size(l, l_t, min_lattice_l):
    grid_size = []
    if l_t in ["Cubic", "Tetragonal", "Orthorhombic"]:
        for i in l:
            if min_lattice_l <= i < 4:
                grid_size.append(4)
            elif 4 <= i < 12:
                grid_size.append(8)
            elif 12 <= i < 20:
                grid_size.append(12)
            elif 20 <= i < 30:
                grid_size.append(16)
            elif 30 <= i < 45:
                grid_size.append(20)
            elif 45 <= i < 60:
                grid_size.append(40)
            elif 60 <= i < 100:
                grid_size.append(50)
            elif 100 <= i < 180:
                grid_size.append(60)
            elif 180 <= i < 240:
                grid_size.append(80)

            else:
                raise ValueError(f"晶格长度过长: {i}, 超出允许范围")
            continue
    elif l_t in ["Hexagonal"]:
        for j in range(3):
            if j in [0, 1]:
                if min_lattice_l <= l[j] < 5:
                    grid_size.append(3)
                elif 5 <= l[j] < 8:
                    grid_size.append(6)
                elif 8 <= l[j] < 12:
                    grid_size.append(9)
                elif 12 <= l[j] < 20:
                    grid_size.append(18)
                elif 25 <= l[j] < 40:
                    grid_size.append(24)
                elif 40 <= l[j] < 60:
                    grid_size.append(30)
                elif 60 <= l[j] <= 100:
                    grid_size.append(60)
                elif 100 <= l[j] < 180:
                    grid_size.append(90)
                elif 180 <= l[j] < 240:
                    grid_size.append(120)
                else:
                    raise ValueError(f"六角晶格的A,B边长度过长: {l[j]}, 超出允许范围")
            elif j == 2:
                if min_lattice_l <= l[j] < 5:
                    grid_size.append(4)
                elif 5 <= l[j] < 14:
                    grid_size.append(8)
                elif 14 <= l[j] < 19:
                    grid_size.append(12)
                elif 19 <= l[j] < 25:
                    grid_size.append(20)
                elif 25 <= l[j] < 40:
                    grid_size.append(24)
                elif 40 <= l[j] < 60:
                    grid_size.append(28)
                elif 60 <= l[j] <= 100:
                    grid_size.append(32)
                elif 100 <= l[j] < 180:
                    grid_size.append(60)
                elif 180 <= l[j] < 240:
                    grid_size.append(80)
                else:
                    raise ValueError(f"晶格长度过长: {l[j]}, 超出允许范围")

    print(f'The Grid Size of Lattice {l_t} {l} is {grid_size} ')
    return grid_size


def generate_phase_name(atom_dict):
    phase = ''
    for specie in sorted(atom_dict.keys()):  # 按键排序
        num = atom_dict[specie]
        phase += specie
        phase += str(num)
    return phase


def generate_orthorhombic_lattices(cell_volume_min, cell_volume_max, scaling_step, min_lattice, compound_type):
    min_length = min_lattice
    ori_lattices = []
    lattices = []
    lattice_types = []
    max_length = 100

    for a in np.arange(min_length, max_length, scaling_step):
        if a ** 3 > cell_volume_max:
            break

        first_b_exceeded = False

        for b in np.arange(max(a-10, min_lattice), 2.0 * a + scaling_step, scaling_step * 2):

            current_vol = a * b * a
            if current_vol > cell_volume_max:
                if b == a:
                    first_b_exceeded = True  # 第一个b就超限，后续a更大，直接终止a循环
                break  # 当前及后续b更大，终止b循环

            # 检查当前b的最大可能体积（c=max_length）是否小于min
            max_possible_vol = a * b * max_length
            if max_possible_vol < cell_volume_min:
                continue  # 当前b无法满足，跳过

            # 处理c循环
            c_exceeded = False
            # for c in np.arange(min_lattice, (10 * a) + scaling_step, scaling_step * 2):
            for c in np.arange(max(a-5, min_lattice), (10 * a) + scaling_step, scaling_step * 2):
                volume = a * b * c
                if volume > cell_volume_max:
                    c_exceeded = True
                    break  # 后续c更大，终止c循环
                elif volume < cell_volume_min:
                    continue  # 继续尝试更大的c
                else:
                    tmp_lattice = [a, b, c]
                    ori_lattices.append(tmp_lattice)
                if c_exceeded:
                    break  # 当前b的后续c更大，终止b循环

        # 若第一个b就超限，直接终止a循环
        if first_b_exceeded:
            break

    # 判断是否需要进行晶格简化
    if len(ori_lattices) > 10:
        lattices = lattices_reduce(ori_lattices=ori_lattices)
    else:
        lattices = ori_lattices

    for i, lattice in enumerate(lattices):
        if lattice[0] == lattice[1] == lattice[2]:
            # Cubic>>Tetragonal>>Orthorhombic
            lattice_types.append("Cubic")

        elif lattice[0] == lattice[1] != lattice[2]:
            # Tetragonal>>Orthorhombic
            lattice_types.append("Tetragonal")

        elif lattice[0] != lattice[1] and lattice[0] != lattice[2]:
            lattice_types.append("Orthorhombic")

        elif lattice[0] == lattice[2] != lattice[1]:
            lattice_types.append("Orthorhombic")

        elif lattice[0] != lattice[1] == lattice[2]:
            lattice_types.append("Orthorhombic")

    for lattice, lattice_type in zip(lattices, lattice_types):
        print(f"{lattice} : {lattice_type}")

    return lattices, lattice_types


def generate_hexagonal_lattices(cell_volume_min, cell_volume_max, scaling_step, min_lattice, compound_type):
    lattices = []
    lattice_types = []
    min_length = min_lattice
    max_length = 100
    for a in np.arange(min_length, max_length, scaling_step):
        for c in np.arange(a, (10 * a) + scaling_step, scaling_step * 2):
            volume = (np.sqrt(3) / 2) * (a ** 2) * c
            if cell_volume_min <= volume <= cell_volume_max:  # 检查体积是否在范围内
                lattices.append([a, a, c])
                print([a, a, c])
            else:
                continue

    if len(lattices) >= 8:
        lattices = lattices_reduce(ori_lattices=lattices)

    for lattice in lattices:
        lattice_types.append("Hexagonal")

    for lattice, lattice_type in zip(lattices, lattice_types):
        print(f"{lattice} : {lattice_type}")

    return lattices, lattice_types


if __name__ == '__main__':
    stoichiometric_dictionary = {"Al": 78,"Ru":24}
    min_volume_f = 1.1
    max_volume_f = 1.6
    scale_step = 1.0
    minimal_lattice = 3.0
    comp_type = "alloy"
    # Determine the Cell Volume Range
    v_min, v_max = set_cell_volume(atom_dict=stoichiometric_dictionary, min_volume_factor=min_volume_f,
                                   max_volume_factor=max_volume_f,
                                   compound_type=comp_type)

    print(f'v_min:{v_min}')
    print(f'v_max:{v_max}')
    orthorhombic_lattices, orthorhombic_lattice_types = generate_orthorhombic_lattices(cell_volume_min=v_min,
                                                                                       cell_volume_max=v_max,
                                                                                       scaling_step=scale_step,
                                                                                       min_lattice=minimal_lattice,
                                                                                       compound_type=comp_type)

    hexagonal_lattices, hexagonal_lattice_types = generate_hexagonal_lattices(cell_volume_min=v_min,
                                                                              cell_volume_max=v_max,
                                                                              scaling_step=scale_step,
                                                                              min_lattice=minimal_lattice,
                                                                              compound_type=comp_type)
