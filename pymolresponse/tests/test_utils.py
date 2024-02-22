import unittest

from pymolresponse import utils


class SplitterTest(unittest.TestCase):
    def test_default(self) -> None:
        """Does the splitter remove empty fields by default properly?"""
        fixed_splitter = utils.Splitter((4, 3, 5, 6, 10, 10, 10, 10, 10, 10))
        line_full = "  60  H 10  s        0.14639   0.00000   0.00000  -0.00000  -0.00000   0.00000"
        line_truncated = "   1  C 1   s       -0.00000  -0.00000   0.00000"
        ref_full = [
            "60",
            "H",
            "10",
            "s",
            "0.14639",
            "0.00000",
            "0.00000",
            "-0.00000",
            "-0.00000",
            "0.00000",
        ]
        ref_truncated = ["1", "C", "1", "s", "-0.00000", "-0.00000", "0.00000"]
        tokens_full = fixed_splitter.split(line_full)
        tokens_truncated = fixed_splitter.split(line_truncated)
        self.assertEqual(ref_full, tokens_full)
        self.assertEqual(ref_truncated, tokens_truncated)

    def test_no_truncation(self) -> None:
        """Does the splitter return even the empty fields when asked?"""
        fixed_splitter = utils.Splitter((4, 3, 5, 6, 10, 10, 10, 10, 10, 10))
        line = "   1  C 1   s       -0.00000  -0.00000   0.00000"
        ref_not_truncated = ["1", "C", "1", "s", "-0.00000", "-0.00000", "0.00000", "", "", ""]
        tokens_not_truncated = fixed_splitter.split(line, truncate=False)
        self.assertEqual(ref_not_truncated, tokens_not_truncated)
