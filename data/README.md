# Data

`effective_pair/B-B_maincluster_1p65_rmin1p50.csv` is the public B-B
effective-pair table used by the reference B16 calculation.

`sigesn_representative_structures/` contains the two relaxed Ge5SiSn2
configurations shown in Fig. 5c,d of the associated article.

The lattice/grid entries are generated locally with:

```bash
python tools/prepare_entries.py \
  --config configs/b16_rmin1p50.yml \
  --summary results/b16_input_summary.json
```

Generated entry, grid, and alpha-cache files are intentionally ignored by
Git because they are deterministic intermediate data.
