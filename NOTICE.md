# Third-Party Notices & Citation Request

Trace2Fold code, evaluation harness, and bundled public benchmark definitions are released under the MIT License (see `LICENSE`). If you use Trace2Fold, its starter targets, its scoring implementation, or its published benchmark scores in a paper, technical report, or public leaderboard, please cite the project (see `CITATION.cff`).

## Optional Third-Party Models & Tools

The optional folding, backbone-generation, inverse-folding control, and structural-novelty adapters under `backends/` interface with independent third-party models and checkpoints. Those models and weights are **not** bundled in this repository and remain subject to their respective upstream licenses:

- **ESMFold2-Fast** (`biohub/ESMFold2-Fast`) — Subject to the Biohub / ESM model license terms on Hugging Face.
- **Boltz-2** (`boltz-community/boltz-2`, Boltz v2.2.1) — MIT License (code and weights; verify current upstream terms at `https://github.com/jwohlwend/boltz`).
- **Apple SimpleFold 3B** (`apple/ml-simplefold`) — Research-only Apple ML Model License (`LICENSE_MODEL` at `https://github.com/apple/ml-simplefold`). Running `backends/simplefold_modal.py` requires explicit `--acknowledge-research-license`.
- **ProteinMPNN** (`dauparas/ProteinMPNN`) — MIT License.
- **RFD4-Proteina** (`RosettaCommons/RFD4-Proteina`) — Subject to the upstream RosettaCommons / NVIDIA license terms.
- **Genie 3** (`aqlaboratory/genie3`, `yeqinglin/genie3`) — Subject to the upstream AlQuraishi Laboratory license terms.
- **Foldseek** & **US-align** — GPL-3.0 / BSD-style academic tools used for optional PDB100 novelty screening.
- **Experimental Calibration Structures (`data/known/`)** — Derived from public RCSB PDB deposits (`1L2Y`, `1TRZ`, `1QY3`, `1I3V`) under CC0 / PDB public-domain data terms.
