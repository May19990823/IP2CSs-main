from functools import lru_cache
from numbers import Integral

from pyxtal.symmetry import Group


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("%s must be a positive integer" % name)
    value = int(value)
    if value <= 0:
        raise ValueError("%s must be positive" % name)
    return value


def _normalise_atom_counts(atom_counts):
    if not hasattr(atom_counts, "items"):
        raise ValueError("atom_counts must be a mapping")
    normalised = {}
    for specie, count in atom_counts.items():
        count = _positive_int("atom count", count)
        normalised[str(specie)] = count
    if not normalised:
        raise ValueError("atom_counts must not be empty")
    return normalised


@lru_cache(maxsize=None)
def wyckoff_multiplicities(space_group):
    """Return unique Wyckoff multiplicities for a space group."""
    space_group = _positive_int("space_group", space_group)
    group = Group(space_group)
    return tuple(sorted({
        int(wp.multiplicity)
        for wp in group.Wyckoff_positions
        if int(wp.multiplicity) > 0
    }, reverse=True))


def find_wyckoff_combination(space_group, atom_count):
    """Find one multiplicity combination that sums to atom_count.

    Multiplicities are treated as reusable. This is intentionally permissive:
    independent atoms can occupy different orbits with the same Wyckoff
    multiplicity, especially for variable-coordinate positions.
    """
    atom_count = _positive_int("atom_count", atom_count)
    multiplicities = wyckoff_multiplicities(space_group)
    reachable = [False] * (atom_count + 1)
    previous = [None] * (atom_count + 1)
    reachable[0] = True

    for total in range(atom_count + 1):
        if not reachable[total]:
            continue
        for multiplicity in multiplicities:
            next_total = total + multiplicity
            if next_total <= atom_count and not reachable[next_total]:
                reachable[next_total] = True
                previous[next_total] = (total, multiplicity)

    if not reachable[atom_count]:
        return None

    combination = []
    total = atom_count
    while total > 0:
        last_total, multiplicity = previous[total]
        combination.append(multiplicity)
        total = last_total
    return combination


def is_wyckoff_feasible(space_group, atom_counts):
    """Return True if every species count can be composed from multiplicities."""
    atom_counts = _normalise_atom_counts(atom_counts)
    return all(
        find_wyckoff_combination(space_group, count) is not None
        for count in atom_counts.values()
    )


def wyckoff_feasibility_details(space_group, atom_counts):
    atom_counts = _normalise_atom_counts(atom_counts)
    details = {
        "space_group": int(space_group),
        "multiplicities": list(wyckoff_multiplicities(space_group)),
        "species": {},
        "feasible": True,
    }
    for specie, count in atom_counts.items():
        combination = find_wyckoff_combination(space_group, count)
        details["species"][specie] = {
            "atom_count": int(count),
            "combination": combination,
            "feasible": combination is not None,
        }
        if combination is None:
            details["feasible"] = False
    return details


def filter_space_groups_by_wyckoff(space_groups, atom_counts, return_rejected=False):
    kept = []
    rejected = []
    for space_group in space_groups:
        details = wyckoff_feasibility_details(space_group, atom_counts)
        if details["feasible"]:
            kept.append(int(space_group))
        else:
            rejected.append({
                "space_group": int(space_group),
                "reason": "wyckoff_multiplicity_infeasible",
                "wyckoff": details,
            })
    if return_rejected:
        return kept, rejected
    return kept
