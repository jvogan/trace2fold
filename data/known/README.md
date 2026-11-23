# Experimental calibration targets

These four experimental PDB fixtures (`1L2Y`, `1TRZ`, `1QY3`, and `1I3V`) are bundled public calibration structures used by the unit test suite (`tests/test_benchmark.py`) to verify prompt generation, multi-chain and disulfide templates, masked-loop alignment, and proper-rotation (`det(R) = +1`) Kabsch scoring. Prompt exports strip PDB identifiers and reveal amino-acid identities only at task-fixed template positions.

| Track | Experimental reference | Observed residues | Tier |
|---|---|---:|---|
| Trp-cage | TC5b, chain A, NMR model 1 | 20 | Easy |
| Mature insulin | Human insulin chains A and B from the crystallographic T/R dimer | 21 + 30 | Medium |
| GFP | Precyclization R96A GFP variant, chain A | 227 | Hard |
| Masked 17-residue loop | Llama VHH A52, chain A | 129, including 17 editable | Hard |

- **Trp-cage.** The 20-residue compact, single-chain fold provides a small positive control. The deposit has 38 solution NMR conformers. Model 1 is the fixed target because the deposit does not specify a representative conformer. [PDB 1L2Y](https://www.rcsb.org/structure/1L2Y) · [Neidigh et al.](https://doi.org/10.1038/nsb798)
- **Mature insulin.** The two complete mature chains remain separate. The retained protomer has one A-chain disulfide (A6–A11) and two interchain disulfides (A7–B7 and A20–B19). All six cysteines are fixed. Other crystal protomers, ions and solvent are outside this fixture. [PDB 1TRZ](https://www.rcsb.org/structure/1TRZ) · [Ciszak and Smith](https://doi.org/10.1021/bi00172a030)
- **GFP.** The complete experimentally resolved barrel and central helix span author residues A3–A229. The first two deposited residues have no coordinates. The R96A construct also contains F64L, S65T, F99S, M153T and V163A relative to the reference reported in the deposit. The precyclization Thr65, Tyr66 and Gly67 have separate canonical residues and backbone atoms. They map to flattened positions 63–65. [PDB 1QY3](https://www.rcsb.org/structure/1QY3) · [Barondeau et al.](https://doi.org/10.1073/pnas.2133463100)
- **Masked 17-residue loop.** SAbDab2 places CDR-H3 at author positions A102–A118. The deposited domain A1–A129 has a complete backbone and a framework disulfide A22–A101. The contestant fills the loop, and the grader aligns the fixed framework and scores the editable residues. This example is unpaired. The SAbDab2 search found no suitable counterpart with the same framework and loop length and a different loop sequence. Both copies in the 1I3V crystal have the same sequence. [PDB 1I3V](https://www.rcsb.org/structure/1I3V) · [Spinelli et al.](https://doi.org/10.1006/jmbi.2001.4856) · [SAbDab2](https://doi.org/10.1093/nar/gkag1091)

## Mapping and regeneration

Every fixture contains the observed N, Cα, C and O atoms in deposited coordinates, with original chain labels and author residue numbers. It also retains the deposited SSBOND records and sulfur atoms for disulfides within the selected chains. The fixtures have a `TER` record at every chain end. The index's `sequence` concatenates the coordinate-backed sequences in chain order. The positions in `fixed`, `mutable`, `disulfides` and `chromophore_precursors` are one-based indices into that flattened sequence. Each extracted amino acid matches the deposit's SEQRES at the mapped position.

The original PDB downloads and generated fixtures are hashed individually in the index. To regenerate from RCSB PDB downloads from the repository root, run:

```sh
source_dir=./known-source
mkdir -p "$source_dir"
for pdb in 1L2Y 1TRZ 1QY3 1I3V; do
  curl -fsS "https://files.rcsb.org/download/${pdb}.pdb" -o "$source_dir/${pdb}.pdb"
done
python backends/prepare_known.py --source-dir "$source_dir"
```
