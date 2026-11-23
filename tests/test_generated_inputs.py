"""RFD4's coordinate-free template uses the requested exact length in every field."""
import importlib.util
import unittest

if importlib.util.find_spec("modal") is None:
    raise unittest.SkipTest("the Modal SDK is required for generation backend tests")

from backends.generate_shapes import monomer_input


class RfdMonomerInputTests(unittest.TestCase):
    def test_expansion_bounds_with_cif_spacing(self):
        fixture = """_atom_site.label_comp_id                     100-100
_atom_site.annotation_expsegmin_1_residue    100
_atom_site.annotation_expsegmax_1_residue    100
_atom_site.condition_sequence_1_residue      <M>
_rfd_design.sampled_contig   100-100
"""
        output = monomer_input(fixture, 72)
        self.assertIn("_atom_site.annotation_expsegmin_1_residue 72", output)
        self.assertIn("_atom_site.annotation_expsegmax_1_residue 72", output)
        self.assertIn("_rfd_design.sampled_contig   72-72", output)
        self.assertIn("<M>", output)
        self.assertNotIn("100", output)


if __name__ == "__main__":
    unittest.main()
