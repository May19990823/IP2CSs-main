#!/usr/bin/env python3
"""Generate solver-ready entries without starting Gurobi."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.compositional import generate_stoichiometric_info
from Utils.config_manager import Config
from ipcss.LatticeGridEngine import LatticeGridEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    config = Config(config_path=args.config)
    chemical_space = sorted(config.get("Elements"))
    phase = "".join(chemical_space)
    _, formulae = generate_stoichiometric_info(
        species_list=chemical_space,
        stoichiometric_range_list=config.get("StoichiometricRatio"),
        specific_stoichiometric_list=config.get("SpecificStoichiometric"),
    )
    engine = LatticeGridEngine(config.resolved_config)
    summaries = engine.generate_entries(
        EntryDir=config.get("input", "entry"),
        name=phase,
        stoichiometric_dict_list=formulae,
        multiplicities=config.get("Multiplicities"),
        dry_run=False,
    )

    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
