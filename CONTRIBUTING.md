# Contributing to Trace2Fold

This guide specifies the protocol for contributing new model evaluations, structure-predictor adapters, reference controls, and harness improvements to Trace2Fold. Every contribution must keep benchmark comparisons reproducible, zero-tool compliant, and free of holdout leaks.

## Submitting a new model evaluation

Trace2Fold separates local development testing on the 20 public `gen-*` starter shapes (`data/generated/`) from official holdout grading on the 24 private targets (`release-*` and `stage2-*`, committed by SHA-256 hashes in `benchmarks/`).

### Local development evaluation (`gen-*` shapes)

1. Run the pinned `v4.3` prompt (`vectors` on the 11 four-atom `RFD4-Proteina` shapes or `geometry` across all 20 `gen-*` shapes) using `trace2fold prompt` or `trace2fold plan`.
2. Enforce the zero-tool single-turn contract: supply only the prompt text, disable code interpreters, web search, file tools, and multi-turn feedback, and record one reply per target.
3. Extract sequences with the deterministic `length-aware-v2` parser in `trace2fold/trials.py` (responses without an exact-length canonical sequence score `0.00`).
4. Fold extracted sequences in single-sequence, zero-MSA, zero-template mode with both `ESMFold2-Fast` (`seed = 0`) and `Boltz-2` (`msa: empty`, `seed = 0`) under `folds/esmfold2/` and `folds/boltz2/`. Optional 5-seed runs use folding seeds `0` through `4`.
5. Grade predicted folds with `uv run trace2fold grade-v4 --submissions <records> --folds <folds> --twin-map examples/starter_pilot_pairs.json --out <report.json>`, keeping `spacefill-v4` weights unchanged (`0.40 GDT-HA`, `0.25 Cα lDDT`, `0.35 Contact F1`).

### Requesting official 24-shape holdout evaluation

Because holdout coordinates remain private to prevent training contamination, maintainers execute official holdout runs against the SHA-256-committed prompts (`v4.2` for Stage 1 comparability and `v4.3` for Stage 2 and new tracks). Open a GitHub issue or pull request specifying the exact model identifier, provider or open-weight checkpoint revision, reasoning effort setting, temperature, and maximum output tokens, along with local `gen-*` verification results.

## Adding a structure predictor or folding adapter

New folding adapters live under `backends/` and must match the shared artifact contract read by `trace2fold/v4_eval.py`.

| Requirement | Specification |
| --- | --- |
| Input boundary | Accept only `{"items": [{"id": "...", "sequence": "..."}]}` with no target coordinates, MSAs, or templates |
| Output files | Write `<id>.pdb` and `<id>.json` (containing ordered `ca` coordinates, optional `plddt` in `[0, 100]`, optional `ptm` in `[0, 1]`, and SHA-256 hashes) |
| Determinism | Pin model checkpoint revision, diffusion step count, recycle count, and per-sequence random seed |
| Offline unit tests | Add `tests/test_<predictor>_modal.py` covering input validation, output schema parsing, and resume checks with zero remote network calls |
| License notice | Document upstream checkpoint license terms in `NOTICE.md` and require explicit opt-in flags for research-only weights |

## Adding a reference control or specialist baseline

Reference controls establish interpretable ceilings and floors for zero-tool language models. Document the exact input modality of any proposed control (`1D H/E/C`, `Cα`-only, or full 4-atom `N, Cα, C, O` coordinates), the number of sampled draws, and the internal selection rule (such as lowest negative log-likelihood without folding oracle selection).

Never select among multiple candidate sequences by folding them against the target structure. Folding-guided search violates the single-turn zero-tool comparison and must be labeled separately if explored in an ablation.

## Holdout protection and repository hygiene

Holdout targets (`release-*`, `stage2-*`, and `genie-*`) exist to prevent training-set contamination. Pull requests must never commit holdout target coordinates, holdout amino-acid sequences, private API keys, local file paths, or account identifiers.

Before opening a pull request, run the full unit test suite and offline benchmark integrity verifier:

```bash
uv run python -m unittest discover -s tests
uv run trace2fold verify-benchmarks --benchmarks-dir benchmarks
```

Both commands must pass with zero errors before review.
