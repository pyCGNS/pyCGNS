#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import unittest


# --------------------------------------------------
# should not be documented with docstrings
class PATTestCase(unittest.TestCase):
    def setUp(self):
        self.T = None

    def eStr(self, code):
        import CGNS.PAT.cgnserrors as CGE

        return CGE.TAG + CGE.TAG_ERROR + "\[%.3d\].*$" % code

    def genTree(self):
        # should not self reference for test
        # then load tree
        import CGNS.PAT.test.disk

        self.T = CGNS.PAT.test.disk.T

    def test_00Module(self):
        pass

    def test_01Check(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK

        self.assertFalse(CGU.checkName("/name"))
        self.assertFalse(CGU.checkNodeName(3))
        self.assertFalse(CGU.checkName(3))
        self.assertFalse(CGU.checkName(["name"]))
        self.assertFalse(CGU.checkName(""))
        self.assertFalse(CGU.checkName("x" * 33))
        self.assertFalse(CGU.checkName("."))
        self.assertFalse(CGU.checkName(".."))
        self.assertFalse(CGU.checkName("         "))
        self.assertTrue(CGU.checkName("name  test"))
        self.assertTrue(CGU.checkName('name"test"'))
        self.assertTrue(CGU.checkName("name\\test"))
        self.assertTrue(CGU.checkName(" name"))
        self.assertTrue(CGU.checkName("name"))
        self.assertTrue(CGU.checkName("x" * 32))
        self.assertTrue(CGU.checkNodeName(["test", None, [], "Zone_t"]))
        self.assertTrue(CGU.checkNodeName(["  test", None, [], "Zone_t"]))
        self.assertTrue(CGU.checkNodeName(["?test#320", None, [], "Zone_t"]))
        self.assertTrue(CGU.checkNodeName(["test   ", None, [], "Zone_t"]))
        self.assertFalse(CGU.checkName("name  test", strict=True))
        self.assertFalse(CGU.checkName("name\\test", strict=True))
        self.assertFalse(CGU.checkName('name"test"', strict=True))
        self.assertFalse(CGU.checkName(" name", strict=True))
        import string

        clist = list(string.ascii_letters + string.digits + string.punctuation + " ")
        clist.remove("/")
        for c in clist:
            self.assertTrue(CGU.checkName("_" + c))
        ex = CGE.cgnsNameError
        fn = CGU.checkName
        self.assertRaisesRegex(ex, self.eStr(2), CGU.checkNodeName, [], dienow=True)
        self.assertRaisesRegex(ex, self.eStr(22), fn, 3, dienow=True)
        self.assertRaisesRegex(ex, self.eStr(23), fn, "", dienow=True)
        self.assertRaisesRegex(ex, self.eStr(24), fn, "/", dienow=True)
        self.assertRaisesRegex(ex, self.eStr(25), fn, "a" * 33, dienow=True)
        self.assertRaisesRegex(ex, self.eStr(29), fn, ".", dienow=True)
        self.assertRaisesRegex(ex, self.eStr(31), fn, " " * 5, dienow=True)
        self.assertRaisesRegex(
            ex, self.eStr(32), fn, " a" * 5, dienow=True, strict=True
        )
        self.assertRaisesRegex(ex, self.eStr(33), fn, '"', dienow=True, strict=True)
        self.assertRaisesRegex(ex, self.eStr(34), fn, "a  a", dienow=True, strict=True)
        self.genTree()
        self.assertTrue(CGU.checkSameNode(self.T, self.T))
        self.assertFalse(CGU.checkSameNode(self.T, [None, None, [], None]))
        self.assertRaisesRegex(
            CGE.cgnsNodeError,
            self.eStr(30),
            CGU.checkSameNode,
            self.T,
            [None, None, [], None],
            dienow=True,
        )

    def test_02Path(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK

        self.genTree()
        p = "/{Base#1}/{Zone-A}/ZoneBC"
        self.assertEqual(p, CGU.getPathNoRoot(p))
        self.assertEqual(p, CGU.getPathNoRoot("/CGNSTree" + p))
        self.assertEqual("/", CGU.getPathNoRoot("/CGNSTree"))
        self.assertEqual(CGU.getPathNormalize("/./{Base#1}/{Zone-A}/ZoneBC"), p)
        self.assertEqual(CGU.getPathNormalize("/./{Base#1}/{Zone-A}/ZoneBC//"), p)
        self.assertEqual(CGU.getPathToList(p), ["", "{Base#1}", "{Zone-A}", "ZoneBC"])
        self.assertEqual(CGU.getPathToList(p, True), ["{Base#1}", "{Zone-A}", "ZoneBC"])
        self.assertEqual(
            CGU.getPathAsTypes(self.T, p),
            [CGK.CGNSTree_ts, CGK.CGNSBase_ts, CGK.Zone_ts, CGK.ZoneBC_ts],
        )
        zone = ["Zone", None, [["ZoneBC", None, [], CGK.ZoneBC_ts]], CGK.Zone_ts]
        p = "/Zone/ZoneBC"
        self.assertEqual(
            CGU.getPathAsTypes(zone, p, legacy=False), [CGK.Zone_ts, CGK.ZoneBC_ts]
        )
        self.assertEqual(CGU.getPathAsTypes(zone, p, legacy=True), [CGK.ZoneBC_ts])

    def test_02PathNormalizeUpward(self):
        # getPathNormalize is documented as following os.path.normpath, which
        # is used here as the reference. The single intended difference is the
        # POSIX rule keeping a leading '//': a CGNS tree has one single root.
        import itertools
        import os.path
        import CGNS.PAT.cgnsutils as CGU

        def reference(path):
            r = os.path.normpath(path)
            if r.startswith("//") and not r.startswith("///"):
                r = r[1:]
            return r

        tokens = ["", ".", "..", "Base", "Zone"]
        paths = []
        seen = set()
        for size in range(1, 5):
            for combination in itertools.product(tokens, repeat=size):
                for absolute in (True, False):
                    p = ("/" if absolute else "") + "/".join(combination)
                    if p and p not in seen:
                        seen.add(p)
                        paths.append(p)
        for p in paths:
            self.assertEqual(CGU.getPathNormalize(p), reference(p), msg=p)

        # an absolute path never normalizes to a path the library itself rejects
        for p in paths:
            n = CGU.getPathNormalize(p)
            if n.startswith("/") and n != "/":
                self.assertTrue(CGU.checkPath(n), msg=p)

        # spelled out: '..' is clamped at the root of an absolute path...
        self.assertEqual(CGU.getPathNormalize("/.."), "/")
        self.assertEqual(CGU.getPathNormalize("/../Base"), "/Base")
        self.assertEqual(CGU.getPathNormalize("/Base/../../Zone"), "/Zone")
        self.assertEqual(CGU.getPathNormalize("/Base/Sub/../../../../Zone"), "/Zone")
        # ...while a relative path keeps the '..' it cannot resolve
        self.assertEqual(CGU.getPathNormalize("Base/../../Zone"), "../Zone")
        self.assertEqual(CGU.getPathNormalize("../../Zone"), "../../Zone")
        self.assertEqual(CGU.getPathNormalize("../.."), "../..")
        self.assertEqual(CGU.getPathNormalize("../Base/.."), "..")
        # unchanged cases
        self.assertEqual(CGU.getPathNormalize("Base/.."), ".")
        self.assertEqual(
            CGU.getPathNormalize("///Base/././//Zone/../Zone/./ZoneBC//."),
            "/Base/Zone/ZoneBC",
        )

    def test_02PathNormalizeCallers(self):
        # every path helper builds on getPathNormalize, an absolute path
        # overshooting the root has to behave as its clamped equivalent
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnskeywords as CGK

        bc = ["ZoneBC", None, [], CGK.ZoneBC_ts]
        zone = ["Zone", None, [bc], CGK.Zone_ts]
        base = ["Base", None, [zone], CGK.CGNSBase_ts]
        tree = [CGK.CGNSTree_s, None, [base], CGK.CGNSTree_ts]

        p = "/Base/Sub/../../../Base/Zone"  # same node as '/Base/Zone'
        self.assertEqual(CGU.getPathNormalize(p), "/Base/Zone")
        self.assertTrue(CGU.checkPath(CGU.getPathNormalize(p)))
        self.assertIs(CGU.getNodeByPath(tree, p), zone)
        self.assertIs(CGU.getChildrenByPath(tree, p), zone[2])
        self.assertEqual(CGU.getTypeByPath(tree, p), CGK.Zone_ts)
        self.assertEqual(CGU.getPathToList(p), ["", "Base", "Zone"])
        self.assertEqual(CGU.getPathNoRoot(p), "/Base/Zone")
        self.assertEqual(CGU.getPathLeaf(p), "Zone")
        self.assertEqual(CGU.getDepth(p), 2)
        self.assertEqual(CGU.getPathAncestor(p), "/Base")
        self.assertTrue(CGU.hasSameRootPath("/Base", p))
        self.assertEqual(CGU.getPathListCommonAncestor(["/Base/Zone", p]), "/Base/Zone")
        self.assertEqual(
            CGU.stackPathItem("/Base/Sub", "..", "..", "..", "Base", "Zone"),
            "/Base/Zone",
        )

    def test_03NodeStructure(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK

        self.assertEqual(["N1", None, [], "N_t"], CGU.nodeCreate("N1", None, [], "N_t"))
        p = CGU.nodeCreate("N1", None, [], "N_t")
        self.assertEqual(
            ["N1", None, [], "N_t"], CGU.nodeCreate("N1", None, [], "N_t", p)
        )
        self.assertEqual(["N1", None, [], "N_t"], p[2][0])
        n = CGU.nodeCreate("N2", None, [], "N_t", p)
        self.assertNotEqual(["N1", None, [], "N_t"], p[2][1])
        self.assertIs(n, p[2][1])
        self.assertEqual(["N2", None, [], "N_t"], CGU.nodeCopy(n))
        c = CGU.nodeCopy(n)
        self.assertIsNot(n, c)
        cname = "NCopy"
        c = CGU.nodeCopy(n, cname)
        self.assertEqual(c[0], cname)
        del n
        self.assertEqual(c[0], cname)

    def test_04NodeChildren(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnslib as CGL
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK

        A = ["A", None, [], None]
        B = ["B", None, [], None]
        C = ["C", None, [], None]
        A[2] += [B]
        CGU.setAsChild(A, C)
        self.assertTrue(CGU.hasChildName(A, "B"))
        self.assertTrue(CGU.hasChildName(A, "C"))
        self.assertTrue(CGU.checkDuplicatedName(A, "D"))
        self.assertFalse(CGU.checkHasChildName(A, "D"))
        D = CGU.nodeCreate("D", None, [], None, parent=A)
        self.assertFalse(CGU.checkDuplicatedName(A, "D"))
        self.assertTrue(CGU.checkHasChildName(A, "D"))
        self.assertFalse(CGU.checkNodeType(C))
        d = {"None": None, "String": "string value", "Integer": 10, "Float": 1.4}
        for n, v in d.items():
            CGL.newDataArray(A, n, v)
        for name in d:
            self.assertTrue(CGU.hasChildName(A, name))

    def test_05NodeValue(self):
        import numpy
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK

        self.genTree()
        v = CGU.getValueByPath(self.T, "/CGNSTree/CGNSLibraryVersion")
        self.assertEqual(v, numpy.array(3.2))
        n = [
            "ZoneType",
            numpy.array(
                ["S", "t", "r", "u", "c", "t", "u", "r", "e", "d"], dtype="S", order="C"
            ),
            [],
            "ZoneType_t",
        ]
        self.assertTrue(CGU.stringValueMatches(n, "Structured"))
        # set*AsArray
        self.assertEqual(
            CGU.setStringAsArray("Structured").tostring().decode("ascii"), "Structured"
        )
        self.assertEqual(CGU.setIntegerAsArray(1), numpy.array(1, dtype="int32"))
        self.assertTrue(
            (
                CGU.setIntegerAsArray(1, 2, 3) == numpy.array([1, 2, 3], dtype="int32")
            ).all()
        )
        self.assertEqual(CGU.setLongAsArray(1), numpy.array(1, dtype="int64"))
        self.assertTrue(
            (CGU.setLongAsArray(1, 2, 3) == numpy.array([1, 2, 3], dtype="int64")).all()
        )
        self.assertEqual(CGU.setFloatAsArray(1), numpy.array(1, dtype="float32"))
        self.assertTrue(
            (
                CGU.setFloatAsArray(1, 2, 3) == numpy.array([1, 2, 3], dtype="float32")
            ).all()
        )
        self.assertEqual(CGU.setDoubleAsArray(1), numpy.array(1, dtype="float64"))
        self.assertTrue(
            (
                CGU.setDoubleAsArray(1, 2, 3) == numpy.array([1, 2, 3], dtype="float64")
            ).all()
        )
        self.assertTrue(
            (
                CGU.setDoubleAsArray(tuple(range(10, 1010, 10)))
                == numpy.array(tuple(range(10, 1010, 10)), dtype="float64")
            ).all()
        )
        n = ["ZoneType", None, [], "ZoneType_t"]
        # set*ByPath
        self.assertTrue(
            CGU.stringValueMatches(
                CGU.setStringByPath(n, "/", "Structured"), "Structured"
            )
        )
        self.assertEqual(
            CGU.setIntegerByPath(n, "/", 1)[1], numpy.array(1, dtype="int32")
        )
        self.assertTrue(
            numpy.array_equal(
                CGU.setIntegerByPath(n, "/", 1, 2, 3)[1],
                numpy.array([1, 2, 3], dtype="int32"),
            )
        )
        self.assertEqual(CGU.setLongByPath(n, "/", 1)[1], numpy.array(1, dtype="int64"))
        self.assertTrue(
            numpy.array_equal(
                CGU.setLongByPath(n, "/", 1, 2, 3)[1],
                numpy.array([1, 2, 3], dtype="int64"),
            )
        )
        self.assertEqual(
            CGU.setFloatByPath(n, "/", 1)[1], numpy.array(1, dtype="float32")
        )
        self.assertTrue(
            numpy.array_equal(
                CGU.setFloatByPath(n, "/", 1, 2, 3)[1],
                numpy.array([1, 2, 3], dtype="float32"),
            )
        )
        self.assertEqual(
            CGU.setDoubleByPath(n, "/", 1)[1], numpy.array(1, dtype="float64")
        )
        self.assertTrue(
            numpy.array_equal(
                CGU.setDoubleByPath(n, "/", 1, 2, 3)[1],
                numpy.array([1, 2, 3], dtype="float64"),
            )
        )
        self.assertTrue(
            numpy.array_equal(
                CGU.setDoubleByPath(n, "/", range(10, 1010, 10))[1],
                numpy.array([range(10, 1010, 10)], dtype="float64"),
            )
        )

    def test_06NodeType(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK

        A = ["A", None, [], None]
        B = ["B", None, [], None]
        C = ["C", None, [], None]
        D = ["D", None, [], None]
        A[2] += [B]
        A[2] += [C]
        C[3] = "UnknownType_t"
        self.assertFalse(CGU.checkNodeType(C))
        self.assertFalse(CGU.checkNodeType(C, [CGK.CGNSBase_ts, CGK.Zone_ts]))
        self.assertFalse(CGU.checkNodeType(C, CGK.CGNSBase_ts))
        C[3] = CGK.CGNSBase_ts
        self.assertTrue(CGU.checkNodeType(C))
        self.assertTrue(CGU.checkNodeType(C, [CGK.CGNSBase_ts, CGK.Zone_ts]))
        self.assertTrue(
            CGU.checkNodeType(C, [CGK.Zone_ts, CGK.CGNSBase_ts, CGK.Zone_ts])
        )
        self.assertTrue(CGU.checkNodeType(C, CGK.CGNSBase_ts))

    def test_07NodeCompliance(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK
        import numpy

        n = CGU.nodeCreate("Base", numpy.array([3, 3]), [], CGK.CGNSBase_ts)
        z = CGU.nodeCreate("ReferenceState", None, [], CGK.ReferenceState_ts, parent=n)
        self.assertTrue(CGU.checkNodeCompliant(n, None))
        self.assertTrue(CGU.checkNodeCompliant(z, None))
        self.assertTrue(CGU.checkNodeCompliant(z, n, dienow=True))
        self.assertFalse(CGU.checkNodeCompliant(None, n))

    def test_08NodeRetrieval(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK

        self.genTree()
        p1 = "/CGNSTree/{Base#1}/{Zone-B}"
        self.assertEqual(CGU.getNodeByPath(self.T, p1)[0], "{Zone-B}")
        p2 = "/CGNSTree/.///{Base#1}/{Zone-B}/../{Zone-A}"
        self.assertEqual(CGU.getNodeByPath(self.T, p2)[0], "{Zone-A}")
        p3 = "/{Base#1}/{Zone-A}/ZoneBC"
        n3 = CGU.getNodeByPath(self.T, p3)
        self.assertEqual(n3[0], "ZoneBC")
        self.assertEqual(CGU.getPathFromNode(self.T, n3), p3)
        c1 = "{BC-2}"
        n31 = CGU.getNodeByPath(n3, c1)
        self.assertEqual(n31[0], "{BC-2}")
        self.assertEqual(CGU.getPathFromNode(n3, n31), "/" + c1)
        c2 = "./{BC-1}"
        self.assertEqual(CGU.getNodeByPath(n3, c2)[0], "{BC-1}")
        filter = "/.*/.*/Zone.*"
        v1 = ["/{Base#1}/{Zone-B}/ZoneBC", "/{Base#1}/{Zone-B}/ZoneGridConnectivity"]
        self.assertEqual(CGU.getPathByNameFilter(self.T, filter)[3:5], v1)
        filter = "/.*/.*/.*/GridConnectivity.*"
        v2 = "/{Base#1}/{Zone-D2}/ZoneGridConnectivity/{CT-D2-C}"
        self.assertEqual(CGU.getPathByTypeFilter(self.T, filter)[-2], v2)
        t = CGK.CGNSBase_ts
        b = CGU.getAncestorByType(self.T, n3, t)
        self.assertEqual(b[3], t)
        self.assertIsNone(CGU.getAncestorByType(self.T, b, CGK.Zone_ts))
        t = CGK.Zone_ts
        self.assertEqual(CGU.getAncestorByType(b, n3, t)[3], t)
        for p in CGU.getPathsByTypeSet(self.T, [CGK.BC_ts]):
            node = CGU.getNodeByPath(self.T, p)
            self.assertEqual(p, CGU.getPathFromNode(self.T, node))
        res = CGU.getPathsByTypeOrNameList(self.T, ["Zaza"])
        self.assertEqual(res, [])

    def test_09NodeDelete(self):
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnserrors as CGE
        import CGNS.PAT.cgnskeywords as CGK
        import numpy

        n = CGU.nodeCreate("Base", numpy.array([3, 3]), [], CGK.CGNSBase_ts)
        r = CGU.nodeCreate("ReferenceState", None, [], CGK.ReferenceState_ts, parent=n)
        d = CGU.nodeCreate("Data", numpy.array([3.14]), [], CGK.DataArray_ts, parent=r)
        self.assertIsNotNone(CGU.hasChildName(r, "Data"))
        CGU.nodeDelete(n, d)
        self.assertIsNone(CGU.hasChildName(r, "Data"))
        d = CGU.nodeCreate("DataZ", numpy.array([3.14]), [], CGK.DataArray_ts, parent=r)
        self.assertIsNotNone(CGU.hasChildName(r, "DataZ"))
        CGU.nodeDelete(r, d)
        self.assertIsNone(CGU.hasChildName(r, "DataZ"))
        CGU.nodeDelete(r, d)
        self.assertIsNone(CGU.hasChildName(r, "DataZ"))
        n = CGU.nodeCreate("Base", numpy.array([3, 3]), [], CGK.CGNSBase_ts)
        r = CGU.nodeCreate("ReferenceState", None, [], CGK.ReferenceState_ts, parent=n)
        d = CGU.nodeCreate("Data", numpy.array([3.14]), [], CGK.DataArray_ts, parent=r)
        self.assertIsNotNone(CGU.getNodeByPath(n, "/Base/ReferenceState/Data"))
        CGU.nodeDelete(n, d)
        self.assertIsNone(CGU.hasChildName(r, "Data"))
        n = CGU.nodeCreate("Base", numpy.array([3, 3]), [], CGK.CGNSBase_ts)
        r = CGU.nodeCreate("ReferenceState", None, [], CGK.ReferenceState_ts, parent=n)
        d = CGU.nodeCreate("Data", numpy.array([3.14]), [], CGK.DataArray_ts, parent=r)
        self.assertIsNotNone(CGU.getNodeByPath(n, "/Base/ReferenceState/Data"))
        CGU.removeChildByName(r, "Data")
        self.assertIsNone(CGU.hasChildName(r, "Data"))


# ---
print("-" * 70 + "\nCGNS.PAT test suite")
suite = unittest.TestLoader().loadTestsFromTestCase(PATTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

# --- last line
