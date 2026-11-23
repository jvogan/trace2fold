# Sequence-only ESMFold2-Fast folding on Modal

ESMFold2-Fast receives only the submitted amino-acid sequences. Each protein chain is passed separately when a sequence has multiple chains. The fold uses the cached `biohub/ESMFold2-Fast` checkpoint, one diffusion sample, seed 0, 20 trunk loops, and 200 diffusion steps. The worker is offline and reads the frozen model and chemical component cache from a read-only Modal volume. The input schema has no fields for target coordinates, templates, or MSAs.

## Batch input

Supply a JSON file with one `items` array. Each entry has an `id` and a `sequence`. IDs are unique even when case is ignored and become output filenames. Sequences use uppercase canonical amino acids. A colon separates chains in a complex.

```json
{
  "items": [
    {"id": "single_chain", "sequence": "NLYIQWLKDGGPSSGRPPPS"}
  ]
}
```

```sh
modal run backends/fold_modal.py \
  --inputs ./folds/inputs.json \
  --output ./folds/outputs
```

The command runs one ephemeral H100 worker and submits groups of up to 32 sequences to reuse the loaded model. Rerun the same command to resume. Complete, validated outputs matching the input sequence and protocol are reused. An existing ID with a different sequence or protocol requires a different output directory or ID. Use one invocation per output directory at a time.

## Artifacts

- `<id>.pdb` contains one structure, with chain IDs `A`, `B`, and onward in input order and residue numbers starting from 1 per chain. Its B-factor column stores pLDDT on a 0–100 scale.
- `<id>.json` contains `id`, the full `sequence`, `chains` with IDs and lengths, and flat `ca` and `plddt` arrays. Array index 0 is the first residue of chain A and subsequent chain arrays follow in the declared order. `ca` is in ångströms at PDB precision and `plddt` is on a 0–100 scale. It includes pTM, ipTM for multi-chain inputs, PDB and sequence SHA-256 hashes, inference time, and model/runtime details.
- `manifest.json` records the batch input hash and counts for completed, reused, failed, and pending items. A failed or interrupted batch preserves completed predictions and exits unsuccessfully.

The backend checks every expected Cα atom and residue identity before accepting each output. Structural comparisons should use the validated output and the intended reference in the scoring process.

The local tests require the Modal SDK and make no remote calls.

```sh
uv run --with modal==1.4.3 python -m unittest discover -s tests -p 'test_fold*.py'
```
