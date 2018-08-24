#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# TESTING RAW CHLONE INTERFACE ***
# - test save first
# - test load

from __future__ import unicode_literals
from __future__ import print_function
from builtins import (bytes, str, range, dict)
import os
import subprocess
import unittest

import numpy

import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib as CGL


def genTrees():
    tree = CGL.newCGNSTree()
    b = CGL.newBase(tree, '{Base}', 2, 3)
    z = CGL.newZone(b, '{Zone}', numpy.array([[5, 4, 0], [7, 6, 0]], order='F'))
    g = CGL.newGridCoordinates(z, 'GridCoordinates')
    d = CGL.newDataArray(g, CGK.CoordinateX_s, numpy.ones((5, 7), dtype='d', order='F'))
    d = CGL.newDataArray(g, CGK.CoordinateY_s, numpy.ones((5, 7), dtype='d', order='F'))
    d = CGL.newDataArray(g, CGK.CoordinateZ_s, numpy.ones((5, 7), dtype='d', order='F'))
    return (tree,)


class MAPTestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.T = genTrees()[0]
            self.L = []
        except ImportError:
            self.T = []
            self.L = []
        self.HDF01 = 'T01.hdf'
        self.HDF02 = 'T02.hdf'

    def unlink(self, filename):
        if os.path.isfile(filename):
            os.unlink(filename)

    def chmod(self, filename, mode):
        if os.path.isfile(filename):
            os.chmod(filename, mode)

    def getDump(self, filename, path, format, fct):
        com = "h5dump -d \"%s/ data%s\" %s" % (path, format, filename)
        r = subprocess.check_output(com, shell=True)
        r = r.decode('ascii')
        d = r.split('\n')[10]
        v = d.split('): ')[1:][0]
        return fct(v)

    def test_000_Module(self):
        pass

    def test_001_Names(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        l = CHLone.FNONE
        l = CHLone.FALL
        l = CHLone.CHLoneException
        l = CHLone.save
        l = CHLone.load

    def test_002_Save_Args(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[910].*", CHLone.save, self.HDF01, self.T, zflag=None)

    def test_003_Load_Args(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[910].*", CHLone.load, self.HDF01, zflag=None)

    def test_004_Args_01(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[911].*", CHLone.load, self.HDF01, flags=[])

    def test_005_Args_02(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[912].*", CHLone.load, self.HDF01, depth=[])

    def test_006_Args_03(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[909].*", CHLone.load, self.HDF01,
                                maxdata=400, threshold=200)

    def test_007_Args_04(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[917].*", CHLone.save, self.HDF01, self.T,
                                links=[['A', 'A', 'A']])

    def test_008_Flags_01(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[908].*", CHLone.load, self.HDF01,
                                flags=CHLone.FDEFAULT, maxdata=400)

    def test_009_Flags_02(self):
        pass  # 907 update + save, deletemissing + update...

    def test_010_Tree_01(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.unlink(self.HDF01)
        self.assertRaisesRegexp(CHLone.CHLoneException, "[906].*",
                                CHLone.save, self.HDF01, None)

    def test_011_Save_01(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.unlink(self.HDF01)
        CHLone.save(self.HDF01, self.T)
        v = self.getDump(self.HDF01, '/{Base}/{Zone}', '[0,1]', int)
        self.assertEqual(v, 7)
        v = self.getDump(self.HDF01, '/{Base}/{Zone}', '[1,1]', int)
        self.assertEqual(v, 6)

    def test_012_Save_02(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.unlink(self.HDF02)
        CHLone.save(self.HDF02, self.T)
        self.chmod(self.HDF02, 0)
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[100].*", CHLone.save, self.HDF02,
                                self.T, flags=CHLone.FUPDATE)
        self.chmod(self.HDF02, 511)
        self.unlink(self.HDF02)

    def test_013_Save_03(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.unlink(self.HDF01)
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[300].*", CHLone.save, self.HDF01,
                                [None, None, self.T, None])

    def test_014_Load(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[100].*", CHLone.load, 'foo.hdf')

    def test_015_Load(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[101].*", CHLone.load, 'chltest.py')

    def test_016_Save(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.assertRaises(CHLone.CHLoneException, CHLone.load, 'foo.hdf')

    def test_017_Save_Update(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        self.unlink(self.HDF01)
        flags = CHLone.FDEFAULT | CHLone.FUPDATE
        self.assertRaisesRegexp(CHLone.CHLoneException,
                                "[100].*", CHLone.save, self.HDF01, self.T, flags=flags)

    def test_018_Load(self):
        from CGNS.MAP import EmbeddedCHLone as CHLone
        CHLone.save(self.HDF01, self.T)
        (t, l, p) = CHLone.load(self.HDF01)


# ---
print('-' * 70 + '\nCGNS.MAP test suite')
suite = unittest.TestLoader().loadTestsFromTestCase(MAPTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

# --- last line
