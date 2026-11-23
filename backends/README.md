# Trace2Fold Backends and Dataset Scripts

The `backends/` directory holds the GPU folding adapters, structural novelty runners, loopback model client, and backbone curation scripts.

## Source checkout versus installed wheel

Installing the `trace2fold` wheel bundles the core Python package, the starter and calibration files in `data/`, and `backends/plain_player.py` (declared in [pyproject.toml](../pyproject.toml)).

Run all other scripts in `backends/` and the archive verifier on `benchmarks/` from a cloned repository checkout. Install the optional `folding` or `curation` extras (`uv sync --extra folding` or `pip install -e ".[folding]"`) before running Modal or curation commands.

## File index

| File | Role | Execution environment | Detailed guide |
| --- | --- | --- | --- |
| `fold_modal.py` | Primary `ESMFold2-Fast` single-sequence folder (`seed = 0`, 20 trunk loops, 200 diffusion steps) | Source checkout + Modal H100 | [FOLD.md](FOLD.md) |
| `boltz_modal.py` | Primary `Boltz-2` single-sequence folder (`msa: empty`, 3 recycles, 200 steps, `seed = 0`) | Source checkout + Modal H100 | [README.boltz.md](README.boltz.md) |
| `simplefold_modal.py` | Third-predictor `Apple SimpleFold 3B` concordance runner (`500` steps, `τ = 0.01`, `seed = 0`) | Source checkout + Modal H100 | [README.simplefold.md](README.simplefold.md) |
| `pdb100_sidecar_modal.py` | `Foldseek` and `US-align` structural novelty screen against `PDB100` | Source checkout + Modal | Inline module docstring |
| `variations_genie_modal.py` | `Genie 3` `Pure-Cα` backbone generation and `PDB100` screen | Source checkout + Modal | Inline module docstring |
| `plain_player.py` | Tool-free OpenAI-compatible Chat Completions client for loopback servers | Installed wheel or source checkout | Used by `trace2fold plan` and `run` |
| `generate_shapes.py` | Generate single-chain backbones with stock `Genie 3` and `RFD4-Proteina` | Source checkout + local GPU or container | Inline module docstring |
| `generate_curate.py` | Curate and freeze matched sequence-masked twin pairs from generation receipts | Source checkout + `curation` extras | Inline module docstring |
| `freeze_backbones.py` | Freeze anonymous 4-atom (`N, Cα, C, O`) coordinates from generated backbones | Source checkout | Inline module docstring |
| `holdout_similarity.py` | Split qualified `Cα` traces by rigid `US-align` structural similarity | Source checkout + `US-align` | Inline module docstring |
| `prepare_known.py` | Freeze the 4 experimental calibration targets from local `wwPDB` files | Source checkout | Inline module docstring |

## Modal cache volumes and configuration

Each GPU folding adapter runs with network access disabled during inference and verifies pinned checkpoint SHA-256 hashes from a Modal Volume before folding.

| Adapter | Default Modal volume | Configuration and notes |
| --- | --- | --- |
| `fold_modal.py` | `esmfold2-models` | Mounts read-only at `/models` with `HF_HUB_OFFLINE=1` for pinned `biohub/ESMFold2-Fast` weights and `ccd.pkl`. |
| `boltz_modal.py` | `trace2fold-boltz2-cache` | Override the volume name with `TRACE2FOLD_BOLTZ_VOLUME`. Verifies `boltz2_conf.ckpt` and `mols.tar` SHA-256 digests at container startup. |
| `simplefold_modal.py` | `trace2fold-simplefold3b-variations-v2` | Stores staged batch receipts and structures under `/campaign` and pinned weights under `/assets`. See [README.simplefold.md](README.simplefold.md). |

## Apple SimpleFold 3B research license

`Apple SimpleFold 3B` model weights ship under a separate upstream non-commercial research license ([NOTICE.md](../NOTICE.md)). Running `backends/simplefold_modal.py` requires passing `--acknowledge-research-license` and, for paid remote execution, `--remote-launch-allowed`.
