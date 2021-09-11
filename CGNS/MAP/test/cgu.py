#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
# TESTING RAW HDF5/C API and H5PY IMPLEMENTATIONS ***
# - test save first
# - test load
import os
import subprocess
import unittest

import numpy


def genTrees():
    import CGNS.PAT.cgnskeywords as CGK
    import CGNS.PAT.cgnslib as CGL

    tree = CGL.newCGNSTree()
    b = CGL.newBase(tree, "{Base}", 2, 3)
    z = CGL.newZone(b, "{Zone}", numpy.array([[5, 4, 0], [7, 6, 0]], order="F"))
    g = CGL.newGridCoordinates(z, "GridCoordinates")
    d = CGL.newDataArray(g, CGK.CoordinateX_s, numpy.ones((5, 7), dtype="d", order="F"))
    d = CGL.newDataArray(g, CGK.CoordinateY_s, numpy.ones((5, 7), dtype="d", order="F"))
    d = CGL.newDataArray(g, CGK.CoordinateZ_s, numpy.ones((5, 7), dtype="d", order="F"))
    return (tree,)


class MAPTestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.T = genTrees()[0]
            self.L = []
        except ImportError:
            self.T = []
            self.L = []
        self.HDF01 = "T01.hdf"
        self.HDF02 = "T02.hdf"

    def unlink(self, filename):
        if os.path.isfile(filename):
            os.unlink(filename)

    def chmod(self, filename, mode):
        if os.path.isfile(filename):
            os.chmod(filename, mode)

    def getDump(self, filename, path, format, fct):
        com = 'h5dump -d "%s/ data%s" %s' % (path, format, filename)
        r = subprocess.check_output(com, shell=True)
        r = r.decode("ascii")
        d = r.split("\n")[10]
        v = d.split("): ")[1:][0]
        return fct(v)

    def void_test_000_Constants(self):
        import CGNS
        import CGNS.MAP
        from CGNS.MAP import flags, flags_set, flags_unset, flags_check

        f1 = flags.NONE
        f2 = flags.ALL
        self.assertEqual(flags.NONE & flags.ALL, flags.NONE)
        self.assertEqual(flags.NONE | flags.ALL & flags.ALL, flags.ALL)
        self.assertEqual(flags.ALL | ~flags.NONE & flags.ALL, flags.ALL)
        self.assertEqual(flags.ALL ^ flags.TRACE & flags.ALL, flags.ALL & ~flags.TRACE)
        l = flags.links.OK
        f = flags_set()
        self.assertEqual(f, flags.DEFAULT)
        f = flags_set(f, flags.TRACE)
        self.assertEqual(f, flags.DEFAULT | flags.TRACE & flags.ALL)
        c = flags_check(f, flags.TRACE)
        self.assertTrue(c)
        f = flags_unset(f, flags.TRACE)
        self.assertEqual(f, flags.DEFAULT & flags.ALL)
        c = flags_check(f, flags.TRACE)
        self.assertFalse(c)

    def void_test_001_Probe(self):
        from CGNS.MAP import probe
        from CGNS.PAT.cgnserrors import CGNSException

        self.assertRaisesRegex(CGNSException, "[900].*", probe, "/Z/ ?/ /U.cgns")

    def void_test_002_Save_Args(self):
        from CGNS.MAP import save, flags
        from CGNS.PAT.cgnserrors import CGNSException

        self.assertRaisesRegex(
            CGNSException, "[910].*", save, self.HDF01, self.T, zflag=None
        )
        self.assertRaisesRegex(
            CGNSException, "[917].*", save, self.HDF01, self.T, links=[["A", "A", "A"]]
        )
        self.unlink(self.HDF01)
        self.assertRaisesRegex(CGNSException, "[906].*", save, self.HDF01, None)
        self.unlink(self.HDF01)
        save(self.HDF01, self.T)
        v = self.getDump(self.HDF01, "/{Base}/{Zone}", "[0,1]", int)
        self.assertEqual(v, 7)
        v = self.getDump(self.HDF01, "/{Base}/{Zone}", "[1,1]", int)
        self.assertEqual(v, 6)
        self.unlink(self.HDF02)
        save(self.HDF02, self.T)
        self.chmod(self.HDF02, 0)
        self.assertRaisesRegex(
            CGNSException, "[100].*", save, self.HDF02, self.T, flags=flags.UPDATE
        )
        self.chmod(self.HDF02, 511)
        self.unlink(self.HDF02)
        self.unlink(self.HDF01)
        self.assertRaisesRegex(
            CGNSException, "[300].*", save, self.HDF01, [None, None, self.T, None]
        )
        self.unlink(self.HDF01)
        flags = flags.DEFAULT | flags.UPDATE
        self.assertRaisesRegex(
            CGNSException, "[100].*", save, self.HDF01, self.T, flags=flags
        )

    def void_test_003_Load_Args(self):
        from CGNS.MAP import load
        from CGNS.PAT.cgnserrors import CGNSException

        self.assertRaisesRegex(CGNSException, "[910].*", load, self.HDF01, zflag=None)
        self.assertRaisesRegex(CGNSException, "[911].*", load, self.HDF01, flags=[])
        self.assertRaisesRegex(CGNSException, "[912].*", load, self.HDF01, depth=[])
        self.assertRaisesRegex(
            CGNSException, "[909].*", load, self.HDF01, maxdata=400, threshold=200
        )
        self.assertRaisesRegex(
            CGNSException, "[908].*", load, self.HDF01, flags=flags.DEFAULT, maxdata=400
        )
        self.assertRaisesRegex(CGNSException, "[100].*", load, "foo.hdf")
        self.assertRaisesRegex(CGNSException, "[101].*", load, "chltest.py")

    def void_test_004_LoadSave(self):
        from CGNS.MAP import load
        from CGNS.PAT.cgnserrors import CGNSException

        save(self.HDF01, self.T)
        (t, l, p) = load(self.HDF01)


# ---
print("-" * 70 + "\nCGNS.MAP test suite")
suite = unittest.TestLoader().loadTestsFromTestCase(MAPTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

# --- last line
