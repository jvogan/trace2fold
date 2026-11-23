# Independent Boltz-2 judge

The Boltz-2 judge receives the same private, opaque-ID sequence list as ESMFold2-Fast. Its input contains only `{"items":[{"id":"…","sequence":"…"}]}`. Use colons to mark separate protein chains. The adapter creates a protein-only Boltz spec with `msa: empty` for every chain. Prediction uses three recycles, 200 diffusion steps, one sample and seed 0 per sequence. The prediction worker has network access disabled and receives no structures, templates, constraints or task descriptions. [Boltz v2.2.1 prediction instructions](https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md)

Run from the repository root with the Modal SDK and a read-only stock Boltz-2 model cache volume (default `trace2fold-boltz2-cache`, configurable with `TRACE2FOLD_BOLTZ_VOLUME`). The cache contains the upstream structure checkpoint and molecule dictionary. The worker checks their SHA-256 hashes before running. [Boltz-2 upstream weights](https://huggingface.co/boltz-community/boltz-2/tree/6fdef46d763fee7fbb83ca5501ccceff43b85607)

```sh
run_dir=./runs/calibration
modal run -m backends.boltz_modal --inputs "$run_dir/fold-inputs.json" --output "$run_dir/boltz2"
```

The output uses the shared fold artifact format. Each completed item has a validated PDB and a JSON sidecar containing its ordered chain IDs, Cα coordinates, residue pLDDT, pTM and multichain ipTM, input and PDB hashes, and model settings. The batch manifest records completed, reused, failed and pending items. A repeated invocation against the same output directory reuses previously validated results for the same input and protocol.

For up to 64 pilot sequences, pass `--workers 8` to run up to eight chunks concurrently. Each H100 processes at most eight sequences per call, running one predictor process at a time. The timeout is 180 seconds per predictor process and 1,800 seconds per call, with up to 600 seconds for startup. The seed is reset for every sequence independently of chunking. [Modal rates](https://modal.com/pricing)

For a larger frozen queue, supply its size with `--max-items N`. The runner uses the same eight-sequence chunks, concurrency limit, per-sequence seed and single manifest for the complete queue.
