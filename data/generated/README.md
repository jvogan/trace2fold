# Public starter development shapes (`gen-*`)

This directory provides 20 frozen single-chain public development backbones (`48`, `72`, and `96` residues) generated with stock
[Genie 3](https://github.com/aqlaboratory/genie3/tree/d77ae5ac04212ff1e8b29b585859a3244c614804)
and native eager
[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina/tree/274c366a0981d680bf3ba79924d99da5e91ae107)
for local prompt inspection, quickstart testing, and 3D visualization (distinct from the 24 private holdout targets at `48`, `80`, `112`, and `176` residues). Eleven of the 20 shapes originate from `RFD4-Proteina` and include frozen 4-atom (`N, Cα, C, O`) coordinates in `backbone_atoms.json` for `vectors` prompts; all 20 shapes support `Pure-Cα` (`geometry`) prompts.

The frozen PDBs contain only the original C-alpha coordinates, rounded to PDB
precision, with chain A, uniform glycine placeholders, residue indices
starting at one, and uniform B-factors. The evaluator index includes
RFD4's co-generated sequence and, for Genie 3, the lowest scoring of four
temperature-0.1 CA-ProteinMPNN draws under that model's own score. The benchmark's
independent folding backend is used to assess the specialist sequences.

## Pairing

Candidates needed finite one-chain coordinates, consecutive C-alpha distances
within 3.2–4.4 Å, no C-alpha distances below 2.9 Å for residues at least four
positions apart, and a C-alpha radius of gyration at most
`3.2 × length^(1/3)` Å. At least 55% of residues needed a C-alpha neighbor
within 8 Å at a sequence separation of six or more residues.

Biotite's P-SEA C-alpha method assigned helix, strand, and coil categories to
both sources. Helical shapes required at least 45% helix and less than 15%
strand. Beta shapes required at least 30% strand and less than 15% helix.
Mixed shapes required at least 15% of each. Pair mates needed equal lengths
and families, at most 20 percentage points of difference in each secondary
fraction, and radii within 22% of the smaller radius. Selected shapes of equal
length differ by at least 3 Å under residue-corresponding C-alpha alignment
and have nonlocal contact-map Jaccard similarity at most 0.55.

## Reproducing the panel

Run each generator once into a fresh artifact root, then run curation with
`numpy`, `biotite==2.0.0`, and a local ProteinMPNN checkout with PyTorch.

```bash
modal run backends/generate_shapes.py --generator genie3 --output "$RAW"
modal run backends/generate_shapes.py --generator rfd4-proteina --output "$RAW"
modal run backends/generate_shapes.py --generator rfd4-proteina-beta --output "$RAW"
python backends/generate_curate.py "$RAW" --proteinmpnn "$PROTEINMPNN/protein_mpnn_run.py" --include-beta
```

The evaluator index is `index.json`. Keep it out of the sequence-generation
prompt. `provenance.json` records model and checkpoint identities, per-target
geometry statistics and hashes, and curation counts. The raw artifact root
holds native outputs, run receipts, and the detailed audit.

## Four-atom representation

`backbone_atoms.json` preserves the original N, Cα, C, and O coordinates for
the 11 RFD4 cases. The original Genie 3 files contain Cα only. The freezer
checks each original file hash and its correspondence to the frozen Cα rows.

```bash
python -m backends.freeze_backbones "$RAW"
```

The v4 vector inputs use a virtual Cβ unit direction derived from each
recorded N–Cα–C geometry in the same coordinate frame as the Cα positions.
The contact diagnostic adds pairs within 8 Å and at least six positions apart.
The masked diagnostic supplies the vector input and a reference sequence with
12–18 interior residues removed. It excludes the first and last three positions
and orders candidates by decreasing nonlocal contact count, increasing distance
to the Cα centroid, and then increasing residue index. The editable count is
one fifth of the chain length rounded down, with a minimum of 12 and a maximum of 18.

For a separate four-atom panel, supply completed, hashed RFD4 artifact lanes to
the curator with Biotite 2.0.0 and NetworkX installed. The ranked dataset uses
rigid Cα alignments at US-align revision `23f14ffcf8647f2dc1c0e01043db3a7f3a661c28`.
Structures are linked if both reported length-normalized TM-scores and both
aligned coverage fractions are at least 0.85. The split excludes every cluster
that touches an earlier single-chain benchmark case and keeps one representative
from each remaining cluster. [US-align source](https://github.com/pylelab/USalign/tree/23f14ffcf8647f2dc1c0e01043db3a7f3a661c28)

The curator publishes a dataset when each requested length and structural
family has the required number of disjoint matched pairs after this split.
Use `--coverage-only` in place of `--dataset-root` to inspect the qualified
counts before publishing. Omitting the alignment arguments in coverage mode
reports the initial geometry screen.

```bash
python -m backends.generate_curate "$RAW" --rfd4-only --rfd4-lanes "$LANES" \
  --lengths 48,80,112 --pairs-per-stratum 8 --dataset-root "$NEW_DATASET" \
  --usalign "$USALIGN" --similarity-cache "$SIMILARITY_CACHE"
```
