# Benchmark Data Archives and Offline Verification

The `benchmarks/` directory stores the canonical, coordinate-free score archives for all published Trace2Fold evaluations, plus 3D overlay coordinates for the 6 public starter development cases. Every table and chart in the root [README.md](../README.md) and the [Interactive Benchmark Explorer](https://jvogan.github.io/trace2fold/) is generated directly from these JSON files.

## Shipped files

| File | Scope | Primary contents | Holdout coordinates or sequences |
| --- | --- | --- | --- |
| `stage-one-scores.json` | Stage 1 (`18` matched shapes at `48`, `80`, `112 aa`) | 14 evaluated model settings, 5 reference baselines, `342` per-target cells, 5-seed arrays, and 3 paired `High`/`XHigh` tables | None (`coordinate_free: true`) |
| `stage-two-scores.json` | Stage 2 (`6` matched shapes at `176 aa`) | 14 evaluated model settings, 5 reference baselines, `114` per-target cells, per-pair splits, and `LOPO` ranges | None (`coordinate_free: true`) |
| `length-scaling-24-shapes.json` | Combined 4-length scaling (`24` matched shapes) | Per-length (`48`, `80`, `112`, `176 aa`) and 24-shape means for 14 models and 5 baselines, weight-sensitivity summaries, and optional `mixed_effort_ladder` | None (`coordinate_free: true`) |
| `variations-and-simplefold.json` | Ablations, third predictor, novelty, and sequence diagnostics | 60-evaluation `Vectors` minus `Pure-Cα` deltas, `Genie 3` pairs, `Apple SimpleFold 3B` concordance (`442` folded cells across all `456` slots plus `128` variation folds), `PDB100` novelty, and `318` sequence diagnostics | None (`coordinate_free: true`) |
| `public-development-3d.json` | Public starter 3D visualization | `6` public `gen-*` target traces, `38` predicted fold overlays, and verbatim `v4.3` / `v4.2` prompts on `gen-l48-02-b` | Public `gen-*` starter shapes only |

## Top-level schema overview

Every file declares `"schema_version": 1` and uses strict RFC 8259 JSON (`NaN` and `Infinity` are rejected).

### `stage-one-scores.json`

Top-level keys: `schema_version`, `stage`, `prompt_version`, `answer_parser`, `scoring_profile`, `weights`, `coordinate_free`, `targets`, `models`, `controls`, `cells`, `paired_efforts`.

- `targets`: 18 target descriptors listing `case`, `length`, `family`, `pair`, `twin`, `generator`, `ordered_label_agreement`, `ca_rmsd`, `contact_jaccard`, and SHA-256 prompt/target commitments.
- `models`: 14 model-effort summaries (`mean_score`, `five_seed_mean`, `best_of_5`, `twin_delta`, `tight_2a_seed0`, `tight_2a_any5`, `tight_2a_all5`, `by_length`, and `by_family`).
- `controls`: 5 reference baselines (`ca-proteinmpnn`, `vanilla-proteinmpnn`, `specialist`, `secondary-structure-rule`, `composition-shuffle`).
- `cells`: `342` per-`(model, target)` and per-`(control, target)` entries with `seed0_score`, `twin_score`, `twin_margin`, `both_within_2a`, `seed_scores`, and per-predictor dictionaries (`esmfold2`, `boltz2`).

### `stage-two-scores.json`

Top-level keys: `schema_version`, `stage`, `prompt_version`, `answer_parser`, `scoring_profile`, `weights`, `coordinate_free`, `targets`, `models`, `controls`, `local_rule_twin_margin`, `cells`, `paired_efforts`.

- `targets`: 6 frozen 176-residue targets (`>= 5.0 Å` pair Cα RMSD) with `hec_agreement_tier` (`strict_70pct` for helical/mixed pairs; `relaxed_65pct` with observed `ordered_label_agreement: 0.6648` for beta Pair 02) and SHA-256 commitments.
- `models`: 14 model summaries with `mean_score`, `five_seed_mean`, `best_of_5`, `twin_delta`, `tight_2a_seed0`, `pair_means`, `lopo_range`, and `mean_24_shapes`.
- `controls`: 5 reference baselines with matching Stage 2 metrics.
- `cells`: `114` per-target evaluation entries across the 6 Stage 2 shapes.

### `length-scaling-24-shapes.json`

Top-level keys: `schema_version`, `lengths` (`[48, 80, 112, 176]`), `shapes_per_length` (`6`), `total_shapes` (`24`), `scoring_profile`, `coordinate_free`, `contestants`, `controls`, `weight_sensitivity`, and optional `mixed_effort_ladder`. Every contestant and control entry stores `by_length` (`"48"`, `"80"`, `"112"`, `"176"`) and `mean_24_shapes` directly.

### `variations-and-simplefold.json`

Top-level keys: `schema_version`, `coordinate_free`, `variations_60call`, `simplefold_3b`, `pdb100_novelty`, `cpu_diagnostics`.

- `variations_60call`: 60-evaluation representation and generator ablation (`vectors_minus_pure_ca` for Stage 1 and Stage 2, `genie3_pure_ca_pairs` for `G3-P01` and `G3-P02`, and per-target `cells`).
- `simplefold_3b`: Third-predictor concordance (`matrix_456_cells` covering `442` folded sequences across all `14` LLM settings and `5` baselines, plus `panel_128_inputs` and `panel_72_sequences`).
- `pdb100_novelty`: `PDB100` (`324,204` chains) `Foldseek` + `US-align` familiarity screens for the 144-target Stage 1 pool and the 6 Stage 2 targets.
- `cpu_diagnostics`: `318` folded-sequence entropy, homopolymer, beta-branched enrichment, and contact-separation recall-decay metrics.

### `public-development-3d.json`

Top-level keys: `schema_version`, `scope`, `cases`, `overlays`, `prompts`. Stores 3D `Cα` coordinates and predicted fold overlays exclusively for the 6 public `gen-*` development targets (`gen-l48-01-a`, `gen-l48-01-b`, `gen-l48-02-b`, `gen-l48-03-b`, `gen-l72-04-a`, `gen-l72-04-b`) together with verbatim `v4.3` and `v4.2` prompt texts on `gen-l48-02-b`.

## Offline verification

After installing package dependencies (`uv sync` or `pip install -e .`, which installs `numpy`), run the built-in verifier from a repository checkout with zero network access. Readers can recompute all table summaries from the released per-cell archives and run full `spacefill-v4` grading on the 20 public `gen-*` shapes; holdout SHA-256 hashes serve as cryptographic commitments because re-folding holdout targets requires the private coordinates.

```bash
uv run trace2fold verify-benchmarks --benchmarks-dir benchmarks
```

The verifier in `trace2fold/v4_eval.py` checks six invariants on every `.json` file:

1. Strict JSON parsing rejecting non-standard numeric constants (`NaN`, `Infinity`).
2. Complete canonical suite coverage (when verifying `benchmarks/` or passing `--strict-suite`): all 5 canonical archive filenames and their expected target and cell counts (`18` targets and `342` cells in Stage 1, `6` targets and `114` cells in Stage 2, `24` total shapes in scaling, `6` cases in `public-development-3d.json`).
3. Numeric range integrity: scores and `pLDDT` in `[0, 100]`, component fractions and `pTM` in `[0, 1]`, non-negative `RMSD`, non-inverted `[lo, hi]` ranges, and signed delta intervals in `[-100, 100]`.
4. Twin-pair symmetry: every declared `twin` link is reciprocal, every `pair` contains two distinct members, and paired twins share identical sequence length.
5. Holdout leak protection: forbids `xyz`, `ca`, `backbone`, `coordinates`, and `>= 40`-residue amino-acid sequence strings outside `public-development-3d.json`, and forbids any `release-*`, `stage2-*`, or `genie-*` coordinates anywhere.
6. Privacy hygiene (generic path/key checks plus salted SHA-256 token digests) and optional manifest SHA-256 verification.

## Quick Python inspection example

```python
import json
from pathlib import Path

stage1 = json.loads(Path("benchmarks/stage-one-scores.json").read_text())
for row in stage1["models"][:5]:
    print(f"{row['label']:26s} score={row['mean_score']:5.2f}  twin_delta={row['twin_delta']:+5.2f}  <=2A={row['tight_2a_seed0']}/18")
```
