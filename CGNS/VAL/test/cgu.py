#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
# TESTING VAL
#
import unittest

import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy as NPY

import importlib
import os
import string
import subprocess


class VALTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_001_Base_write_close(self):
        import CGNS.VAL.suite.run

        CGNS.VAL.suite.run.runall()


# ---
print("-" * 70 + "\nCGNS.VAL test suite")
suite = unittest.TestLoader().loadTestsFromTestCase(VALTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

# --- last line
