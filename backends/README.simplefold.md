# SimpleFold 3B on Modal

Use this adapter for one Apple SimpleFold 3B sample per sequence, with 500 steps, τ 0.01 and seed 0. A single H100 worker loads the pinned models once per attempt. The input is a canonical single-chain sequence. The raw ModelCIF preserves upstream output, including its placeholder B factors. The companion PDB has zero B factors. Confidence and PAE are unavailable. [Apple SimpleFold source and model license](https://github.com/apple/ml-simplefold/blob/c7a5570a6be9f5c695126e27c804e77567209934/LICENSE_MODEL)

## Freeze and plan

Set `RUN` to your local campaign output directory. The parent ledger has 72 rows in an `items` array with `id` and `sequence` fields. `freeze-canary` selects the predeclared 48-, 112- and 176-residue rows in order and binds them to the SHA-256 of the exact parent ledger bytes. The adapter discards extra row fields at the input boundary.

```sh
FULL="$RUN/fold-inputs.json"
SF="$RUN/simplefold"
CANARY="$SF/canary-ledger.json"
PLAN="$SF/modal-plan.json"
CAP=250
uv run --no-project --with modal==1.4.3 python -m backends.simplefold_modal freeze-canary \
  --full-inputs "$FULL" --canary-inputs "$CANARY"
uv run --no-project --with modal==1.4.3 --with gemmi==0.7.3 --with numpy \
  python -m backends.simplefold_modal plan --full-inputs "$FULL" --canary-inputs "$CANARY" \
  --plan-file "$PLAN" --max-spend-usd "$CAP" --acknowledge-research-license
modal run -m backends.simplefold_modal::launch --help
```

Plan validation runs locally before any remote worker starts. Specify `--max-spend-usd` to enforce an upper compute cap across the canary and full stages, with at most one explicit resume per stage within that cap. Execution timeouts are one hour for the canary and 12 hours for the full batch, with a 20-minute startup limit and a 15-minute committed-heartbeat silence limit. [Modal pricing](https://modal.com/pricing), [Modal timeouts](https://modal.com/docs/guide/timeouts)

## Run the canary and verify egress

Check the selected workspace before creating the private Volume. Run each stage attached from one controller. The option `--remote-launch-allowed` explicitly enables the paid worker.

```sh
VOLUME=trace2fold-simplefold3b-variations-v2
modal profile current
modal volume create "$VOLUME"
modal run -m backends.simplefold_modal::launch --stage canary \
  --full-inputs "$FULL" --canary-inputs "$CANARY" --plan-file "$PLAN" --max-spend-usd "$CAP" \
  --acknowledge-research-license --remote-launch-allowed \
  --controller-receipt "$SF/canary-controller.json"

PROFILE="$(modal profile current)"
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE="$PROFILE" \
  uv run --no-project --with modal==1.4.3 --with gemmi==0.7.3 --with numpy \
  python -m backends.simplefold_modal download --stage canary \
  --full-inputs "$FULL" --canary-inputs "$CANARY" --plan-file "$PLAN" --max-spend-usd "$CAP" \
  --acknowledge-research-license --destination "$SF/canary-egress" \
  --controller-receipt "$SF/canary-controller.json"
uv run --no-project --with modal==1.4.3 --with gemmi==0.7.3 --with numpy \
  python -m backends.simplefold_modal verify --stage canary \
  --full-inputs "$FULL" --canary-inputs "$CANARY" --plan-file "$PLAN" --max-spend-usd "$CAP" \
  --acknowledge-research-license --artifacts "$SF/canary-egress" \
  --export "$SF/canary-structures.json" --canary-gate "$SF/canary-gate.json"
```

Canary completion requires all three structures, exact residue identities and counts, complete backbones, finite coordinates, matching CIF/PDB coordinates and SHA-256 evidence. The local verifier also reads each neutral PDB with the SpaceFill atom and Cα parser. The check uses sequence identity and file validity without a target or fit threshold.

## Run the full ledger and close out

Use the same plan, ledger files and spend cap. Pass the canary gate produced by local verification. The controller checks its binding to the remote canary receipt and confirms the prior application has stopped and has zero running containers. Duplicate input hashes reuse already committed artifacts.

```sh
modal run -m backends.simplefold_modal::launch --stage full \
  --full-inputs "$FULL" --canary-inputs "$CANARY" --plan-file "$PLAN" --max-spend-usd "$CAP" \
  --acknowledge-research-license --remote-launch-allowed --canary-gate "$SF/canary-gate.json" \
  --controller-receipt "$SF/full-controller.json"

env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE="$PROFILE" \
  uv run --no-project --with modal==1.4.3 --with gemmi==0.7.3 --with numpy \
  python -m backends.simplefold_modal download --stage full \
  --full-inputs "$FULL" --canary-inputs "$CANARY" --plan-file "$PLAN" --max-spend-usd "$CAP" \
  --acknowledge-research-license --destination "$SF/full-egress" \
  --controller-receipt "$SF/full-controller.json"
uv run --no-project --with modal==1.4.3 --with gemmi==0.7.3 --with numpy \
  python -m backends.simplefold_modal verify --stage full \
  --full-inputs "$FULL" --canary-inputs "$CANARY" --plan-file "$PLAN" --max-spend-usd "$CAP" \
  --acknowledge-research-license --artifacts "$SF/full-egress" \
  --export "$SF/full-structures.json"
```

For monitoring, read `/campaign/<stage>/heartbeat.json` with `modal volume get`, substituting `canary` or `full` for `<stage>`. `download` verifies the Modal application has stopped with zero running containers before and after transfer; if an application is still active, stop its `app_id` from the controller receipt with `modal app stop "$APP_ID"` and re-run `download`.

The `/campaign` snapshot contains the plan, progress receipts, per-ID status, and content-addressed CIF/PDB files, while `egress.json` records the Volume identity and downloaded file hashes. Each transfer requires a fresh destination directory. The `/assets` directory holds the pinned model checkpoints and chemical component dictionary. [Modal Volumes](https://modal.com/docs/guide/volumes)
