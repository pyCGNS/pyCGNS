#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
# IGES reader -v0.0
# marc.poinot@safran.fr
# ??? marks non understandable fixes
import sys
import re
import os
import numpy

import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import CGNS.MAP as CGM
import CGNS.version

doc1 = """
  IGES to CGNS translator
  (part of pyCGNS distribution http://pycgns.github.com)
  pyCGNS v%s
  
  The result of the grep is a file with a proprietatry translation
  from IGES to CGNS.

""" % (
    CGNS.version.__version__
)

doc2 = """
  Exemples:

  cg_iges 110-000.igs 110-000.hdf
  """

import argparse
import re


def parse():
    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] inputfile outputfile",
    )

    pr.add_argument("-d", "--debug", action="store_true", help="trace mode for debug")
    pr.add_argument("files", nargs=argparse.REMAINDER)
    args = pr.parse_args()
    return args


# -----------------------------------------------------------------------------
class IGESentity(object):
    (
        BLANKED,
        VISIBLE,
        INDEPENDANT,
        PHYDEPEND,
        LOGDEPEND,
        ALLDEPEND,
        CONSTRUCTION,
        PARAMETRIC2D,
        LOGICAL,
        OTHER,
        DEFINITION,
        ANNOTATION,
        GEOMETRY,
        TOPDOWN,
        DEFER,
        PROPERTY,
    ) = range(16)
    _directoryentries = {}
    _parameters = {}
    _entities = {}

    @classmethod
    def solve(cls):
        for e in cls._directoryentries:
            if cls._directoryentries[e].pending:
                cls._directoryentries[e].updatereference()

    @classmethod
    def sortedentities(cls):
        k = list(cls._directoryentries)
        k.sort()
        for e in k:
            yield cls._directoryentries[e]

    @classmethod
    def find(cls, index):
        return cls._directoryentries[index]

    @classmethod
    def stat(self):
        print(list(IGESentity._directoryentries))
        print(list(IGESentity._parameters))

    @classmethod
    def allentities(self, code=None):
        if code is None:
            el = []
            for e in IGESentity._entities.values():
                el += e
            return el
        return IGESentity._entities[code]

    def __init__(self, code, text, index, *args):
        self.code = code
        self.text = text
        self.index = index
        if text is "":
            self.text = ENTTABLE[(code, 0)]
        self.pending = False
        self.children = []
        self.childrenid = []
        self.childrenname = {}
        self.refcnt = 0
        self.parseargs(*args)
        IGESentity._directoryentries[self.index[0]] = self
        IGESentity._parameters[self.index[1]] = self
        if self.code not in IGESentity._entities:
            IGESentity._entities[self.code] = []
        IGESentity._entities[self.code] += [self]

    def parseargs(self, args):
        self.parseattributes(*args[:-1])
        self.parseparameters(*args[-1])
        self.args = args

    def parseattributes(self, *data):
        self.a_parameterdata = data[0]
        self.a_structure = data[1]
        self.a_linefont = data[2]
        self.a_level = data[3]
        self.a_view = data[4]
        self.a_matrix = data[5]
        self.a_displaylabel = data[6]
        self.a_status = self.parsestatus("%.8d" % data[7])
        self.a_lineweight = data[8]
        self.a_color = data[9]
        self.a_linecount = data[10]
        self.a_form = data[11]
        self.a_label = data[14]
        self.a_subscript = data[15]

    def parseparameters(self, *data):
        pass

    def parsestatus(self, st):
        if st[0:2] == "00":
            self.s_blank = IGESentity.BLANKED
        else:
            self.s_blank = IGESentity.VISIBLE
        if st[2:4] == "03":
            self.s_depend = IGESentity.ALLDEPEND
        if st[2:4] == "02":
            self.s_depend = IGESentity.LOGDEPEND
        if st[2:4] == "01":
            self.s_depend = IGESentity.PHYDEPEND
        else:
            self.s_depend = IGESentity.INDEPENDANT
        if st[4:6] == "06":
            self.s_use = IGESentity.CONSTRUCTION
        if st[4:6] == "05":
            self.s_use = IGESentity.PARAMETRIC2D
        if st[4:6] == "04":
            self.s_use = IGESentity.LOGICAL
        if st[4:6] == "03":
            self.s_use = IGESentity.OTHER
        if st[4:6] == "02":
            self.s_use = IGESentity.DEFINITION
        if st[4:6] == "01":
            self.s_use = IGESentity.ANNOTATION
        else:
            self.s_use = IGESentity.GEOMETRY
        if st[6:8] == "02":
            self.s_hierarchy = IGESentity.TOPDOWN
        if st[6:8] == "01":
            self.s_hierarchy = IGESentity.DEFER
        else:
            self.s_hierarchy = IGESentity.PROPERTY

    def statusasstring(self):
        s = ""
        if self.s_blank == IGESentity.BLANKED:
            s += "-"
        if self.s_blank == IGESentity.VISIBLE:
            s += "+"
        if self.s_depend == IGESentity.ALLDEPEND:
            s += "*"
        if self.s_depend == IGESentity.PHYDEPEND:
            s += "P"
        if self.s_depend == IGESentity.LOGDEPEND:
            s += "L"
        if self.s_depend == IGESentity.INDEPENDANT:
            s += "-"
        if self.s_use == IGESentity.CONSTRUCTION:
            s += "c"
        if self.s_use == IGESentity.PARAMETRIC2D:
            s += "p"
        if self.s_use == IGESentity.LOGICAL:
            s += "l"
        if self.s_use == IGESentity.OTHER:
            s += "-"
        if self.s_use == IGESentity.DEFINITION:
            s += "d"
        if self.s_use == IGESentity.ANNOTATION:
            s += "a"
        if self.s_use == IGESentity.GEOMETRY:
            s += "g"
        if self.s_hierarchy == IGESentity.PROPERTY:
            s += "-"
        if self.s_hierarchy == IGESentity.TOPDOWN:
            s += "T"
        if self.s_hierarchy == IGESentity.DEFER:
            s += "D"
        return s

    def __repr__(self):
        self.a_status_s = self.statusasstring()
        self.index_s = "%.4d:%.4d" % self.index
        s = "=" * 70 + "\n"
        s += """Entity  #%(index_s)s [%(text)s] (code %(code)d)\n"""
        s += """level : %(a_level)-8s label :%(a_label)-8s refcnt:%(refcnt)d\n"""
        s += """status: %(a_status_s)-8s color :%(a_color)-8s\n"""
        s += """params: %(a_linecount)-8s form  :%(a_form)-8s\n"""
        s = s % self.__dict__
        if self.children:
            s += """children :\n"""
            for e in self.children:
                s += """         : %.4d:%.4d [%s] %s\n""" % (
                    e.index[0],
                    e.index[1],
                    e.text,
                    e.asstring(),
                )
        return s

    def label(self):
        return self.a_label

    def asCGNS(self):
        t = []
        v = CGU.setStringAsArray(self.text)
        t.append(["_text", v, [], "DataArray_t"])
        t.append(["_label", CGU.setStringAsArray(self.a_label), [], "DataArray_t"])
        t.append(["_code", numpy.array([self.code]), [], "DataArray_t"])
        t.append(["_color", numpy.array([self.a_color]), [], "DataArray_t"])
        t.append(["_level", numpy.array([self.a_level]), [], "DataArray_t"])
        t.append(["_view", numpy.array([self.a_view]), [], "DataArray_t"])
        return t

    def remain(self, data):
        if data:
            n = data[0]
            cnt = 1
            if n:
                for ptr in data[1 : 1 + n]:
                    self.addchildid(ptr, "ptr#%.4d" % cnt)
                    cnt += 1
            p = data[n + 1]
            cnt = 1
            if p:
                for prop in data[n + 2 : n + 2 + p]:
                    self.addchildid(prop, "prp#%.4d" % cnt)

    def allchildren(self):
        return self.childrenid

    def haschild(self, id):
        if id in self.childrenid:
            return True
        return False

    def addchildid(self, idx, name):
        if idx:
            self.pending = True
            if idx in IGESentity._parameters:
                e = IGESentity._parameters[idx]
                idx = e.index[0]
            self.childrenid += [idx]
            self.childrenname[idx] = name

    def updatereference(self):
        for c in self.childrenid:
            if not c:
                continue
            try:
                child = IGESentity._directoryentries[c]
            except KeyError as e:
                # fail, index is not DE, try parameters index (not compliant)
                child = IGESentity._parameters[c]
            self.children.append(child)
            child.refcnt += 1

    def asstring(self):
        return ""


# -----------------------------------------------------------------------------
class BSpline(IGESentity):
    def __init__(self, index, *args):
        t = "# Rational B-Spline Curve"
        IGESentity.__init__(self, 126, t, index, *args)

    def parseparameters(self, k, m, p1, p2, p3, p4, *data):
        self.k = k
        self.m = m
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        n = 1 + k - m
        a = n + 2 * m
        self.kn = numpy.array(data[0 : a + 1])
        self.wg = numpy.array(data[a + 1 : a + (k + 1) + 1])
        self.xyz = numpy.array(data[a + k + 2 : a + 1 + 4 * (k + 1)])
        self.v0v1 = numpy.array(data[a + 1 + 4 * (k + 1) : a + 1 + 4 * (k + 1) + 2])
        self.xyzn = numpy.array(data[a + 1 + 4 * (k + 1) + 2 : a + 1 + 4 * (k + 1) + 5])
        self.remain(data[a + 1 + 4 * (k + 1) + 5 :])

    def asCGNS(self):
        t = super(BSpline, self).asCGNS()
        t.append(["k", CGU.setIntegerAsArray(self.k), [], "DataArray_t"])
        t.append(["m", CGU.setIntegerAsArray(self.m), [], "DataArray_t"])
        t.append(["p1", CGU.setIntegerAsArray(self.p1), [], "DataArray_t"])
        t.append(["p2", CGU.setIntegerAsArray(self.p2), [], "DataArray_t"])
        t.append(["p3", CGU.setIntegerAsArray(self.p3), [], "DataArray_t"])
        t.append(["p4", CGU.setIntegerAsArray(self.p4), [], "DataArray_t"])
        t.append(["p4", CGU.setIntegerAsArray(self.p4), [], "DataArray_t"])
        t.append(["knot", self.kn, [], "DataArray_t"])
        t.append(["weight", self.wg, [], "DataArray_t"])
        t.append(["control", self.xyz, [], "DataArray_t"])
        t.append(["v", self.v0v1, [], "DataArray_t"])
        t.append(["normal", self.xyzn, [], "DataArray_t"])
        return t


# -----------------------------------------------------------------------------
class ParametricSplineCurve(IGESentity):
    def __init__(self, index, *args):
        t = "# Parametric Spline Curve"
        IGESentity.__init__(self, 112, t, index, *args)
        self.parseargs(*args)

    def parseparameters(self, ctype, h, ndim, n, *data):
        self.ctype = ctype
        self.h = h
        self.ndim = ndim
        self.n = n
        self.t = numpy.array(data[0 : n + 1])
        self.px = numpy.array(data[n + 1 : n + 1 + 5])
        self.py = numpy.array(data[n + 1 + 5 : n + 1 + 9])
        self.pz = numpy.array(data[n + 1 + 9 : n + 1 + 13])
        # self.remain(data[a+b+4*c+6:])

    def asCGNS(self):
        t = super(ParametricSplineCurve, self).asCGNS()
        t.append(["ctype", CGU.setIntegerAsArray(self.ctype), [], "DataArray_t"])
        t.append(["h", CGU.setIntegerAsArray(self.h), [], "DataArray_t"])
        t.append(["ndim", CGU.setIntegerAsArray(self.ndim), [], "DataArray_t"])
        t.append(["n", CGU.setIntegerAsArray(self.n), [], "DataArray_t"])
        t.append(["t", self.t, [], "DataArray_t"])
        t.append(["px", self.px, [], "DataArray_t"])
        t.append(["py", self.py, [], "DataArray_t"])
        t.append(["pz", self.pz, [], "DataArray_t"])
        return t


# -----------------------------------------------------------------------------
class RationalBSplineSurface(IGESentity):
    def __init__(self, index, *args):
        t = "# Rational B-Spline Surface"
        IGESentity.__init__(self, 128, t, index, *args)
        self.parseargs(*args)

    def parseparameters(self, k1, k2, m1, m2, p1, p2, p3, p4, p5, *data):
        self.k1 = k1
        self.k2 = k2
        self.m1 = m1
        self.m2 = m2
        n1 = 1 + k1 - m1
        n2 = 1 + k2 - m2
        a = n1 + 2 * m1
        b = n2 + 2 * m2
        c = (1 + k1) * (1 + k2)
        self.s1 = numpy.array(data[0 : a + 1])  # ??? why +1
        self.s2 = numpy.array(data[a + 1 : a + b + 2])  # ??? why +1
        self.wg = numpy.array(data[a + b + 2 : a + b + c + 2])
        self.xyz = numpy.array(data[a + b + c + 2 : a + b + 4 * c + 2])
        self.u0u1 = numpy.array(data[a + b + 4 * c + 2 : a + b + 4 * c + 4])
        self.v0v1 = numpy.array(data[a + b + 4 * c + 4 : a + b + 4 * c + 6])
        self.remain(data[a + b + 4 * c + 6 :])

    def asCGNS(self):
        t = super(RationalBSplineSurface, self).asCGNS()
        t.append(["k1", CGU.setIntegerAsArray(self.k1), [], "DataArray_t"])
        t.append(["k2", CGU.setIntegerAsArray(self.k2), [], "DataArray_t"])
        t.append(["m1", CGU.setIntegerAsArray(self.m1), [], "DataArray_t"])
        t.append(["m2", CGU.setIntegerAsArray(self.m2), [], "DataArray_t"])
        t.append(["knot1", self.s1, [], "DataArray_t"])
        t.append(["knot2", self.s2, [], "DataArray_t"])
        t.append(["weight", self.wg, [], "DataArray_t"])
        t.append(["control", self.xyz, [], "DataArray_t"])
        t.append(["u", self.u0u1, [], "DataArray_t"])
        t.append(["v", self.v0v1, [], "DataArray_t"])
        return t


# -----------------------------------------------------------------------------
class PlaneUnbounded(IGESentity):
    def __init__(self, index, *args):
        t = "# Plane:Unbounded"
        IGESentity.__init__(self, 108, t, index, *args)

    def parseparameters(self, c1, c2, c3, c4, ptr, x, y, z, sz, *data):
        self.coefs = numpy.array([c1, c2, c3, c4])
        self.remain(data)

    def asCGNS(self):
        t = super(PlaneUnbounded, self).asCGNS()
        t.append(["coefs", self.coefs, [], "DataArray_t"])
        return t


# -----------------------------------------------------------------------------
class TrimmedParametricSurface(IGESentity):
    def __init__(self, index, *args):
        t = "# Trimmed (Parametric) Surface"
        IGESentity.__init__(self, 144, t, index, *args)

    def parseparameters(self, pts, n1, n2, pt0, *data):
        self.pts = pts
        self.n1 = n1
        self.n2 = n2
        self.pt0 = pt0
        self.addchildid(self.pts, "pts")
        self.addchildid(self.pt0, "pt0")
        for n in range(n2):
            self.addchildid(data[n], "pt#%d" % n)
        self.remain(data[n2:])


# -----------------------------------------------------------------------------
class CompositeCurve(IGESentity):
    def __init__(self, index, *args):
        t = "# Composite Curve"
        IGESentity.__init__(self, 102, t, index, *args)

    def parseparameters(self, n, *data):
        self.n = n
        cnt = 1
        for ptr in data:
            self.addchildid(ptr, "ptr#%.4d" % cnt)
            cnt += 1


# -----------------------------------------------------------------------------
class CurveParametricSurface(IGESentity):
    def __init__(self, index, *args):
        t = "# Curve on a Parametric Surface"
        IGESentity.__init__(self, 142, t, index, *args)

    def parseparameters(self, crtn, sptr, bptr, cptr, pref, *data):
        self.crtn = crtn
        self.sptr = sptr
        self.bptr = bptr
        self.cptr = cptr
        self.pref = pref
        self.addchildid(self.sptr, "sptr")
        self.addchildid(self.bptr, "bptr")
        self.addchildid(self.cptr, "cptr")
        self.remain(data)


# -----------------------------------------------------------------------------
class Property(IGESentity):
    _count = 0

    def __init__(self, index, *args):
        t = "# Property"
        IGESentity.__init__(self, 406, t, index, *args)

    def parseattributes(self, *data):
        IGESentity.parseattributes(self, *data)

    def parseparameters(self, n, *data):
        self.properties = data[0:n]
        if type(data[0]) not in [
            str,
        ]:
            Property._count += 1
            self.property = "P%.4d" % Property._count
        else:
            self.property = data[0]

    def asstring(self):
        print(self.properties)
        if self.properties:
            if type(self.properties == tuple):
                return "%s" % str(self.properties)
            return "%s" % self.properties
        return ""

    def __repr__(self):
        s = IGESentity.__repr__(self) + "property: %s" % self.asstring()
        return s

    def asCGNS(self):
        t = super(Property, self).asCGNS()
        t.append(["property", CGU.setStringAsArray(self.property), [], "DataArray_t"])
        return t


# -----------------------------------------------------------------------------
class Line(IGESentity):
    def __init__(self, index, *args):
        t = "# Line:Bounded (default)"
        IGESentity.__init__(self, 110, t, index, *args)

    def parseparameters(self, x1, y1, z1, x2, y2, z2, *data):
        self.p1 = numpy.array([x1, y1, z1])
        self.p2 = numpy.array([x2, y2, z2])
        self.remain(data)

    def asCGNS(self):
        t = super(Line, self).asCGNS()
        t.append(["pt1", self.p1, [], "DataArray_t"])
        t.append(["pt2", self.p2, [], "DataArray_t"])
        return t


# -----------------------------------------------------------------------------
ENTTABLE = {
    (000, 0): "Null Entity",
    (100, 0): "Circular Arc",
    (102, 0): CompositeCurve,
    (104, 1): "Conic Arc:Ellipse",
    (104, 2): "Conic Arc:Hyperbola",
    (104, 3): "Conic Arc:Parabola",
    (106, 11): "Copius Data:Piecewise Linear Curve (2D)",
    (106, 12): "Copius Data:Piecewise Linear Curve (3D)",
    (106, 20): "Copius Data:Centerline (through points)",
    (106, 21): "Copius Data:Centerline (through circle centers)",
    (106, 31): "Copius Data:Section (Iron, General Use)",
    (106, 32): "Copius Data:Section (Steel)",
    (106, 33): "Copius Data:Section (Bronze, brass, copper, etc.)",
    (106, 34): "Copius Data:Section (Rubber, plastic, etc.)",
    (106, 35): "Copius Data:Section (Titanium, etc.)",
    (106, 36): "Copius Data:Section (Marble, slate, glass, etc.)",
    (106, 37): "Copius Data:Section (Zinc, lead, etc.)",
    (106, 38): "Copius Data:Section (Magnesium, aluminum, etc.)",
    (106, 40): "Copius Data:Witness Line",
    (106, 63): "Copius Data:Simple Closed Planar Curve",
    (108, 0): PlaneUnbounded,
    (108, 1): "Plane:Bounded",
    (110, 0): Line,
    (112, 0): ParametricSplineCurve,
    (114, 0): "Parametric Spline Surface",
    (116, 0): "Point",
    (118, 0): "Ruled Surface:Equal Relative Arc Length",
    (118, 1): "Ruled Surface:Equal Relative Parametric Values",
    (120, 0): "Surface of Revolution",
    (122, 0): "Tabulated Cylinder",
    (124, 0): "Transformation Matrix:Right-handed (default)",
    (124, 1): "Transformation Matrix:Left-handed",
    (126, 0): BSpline,
    (126, 1): "Rational B-Spline Curve - Line",
    (126, 2): "Rational B-Spline Curve - Circular Arc",
    (126, 3): "Rational B-Spline Curve - Elliptical Arc",
    (126, 4): "Rational B-Spline Curve - Parabolic Arc",
    (126, 5): "Rational B-Spline Curve - Hyperbolic Arc",
    (128, 0): RationalBSplineSurface,
    (128, 2): "Rational B-Spline Surface:Right Circular Cylinder",
    (128, 3): "Rational B-Spline Surface:Cone",
    (128, 4): "Rational B-Spline Surface:Sphere",
    (128, 5): "Rational B-Spline Surface:Torus",
    (128, 9): "Rational B-Spline Surface:General Quadratic Surface",
    (130, 0): "Offset Curve",
    (140, 0): "Offset Surface",
    (142, 0): CurveParametricSurface,
    (144, 0): TrimmedParametricSurface,
    (202, 0): "Angular Dimension",
    (206, 0): "Diameter Dimension",
    (202, 0): "General Label",
    (212, 0): "General Note:Simple Note (default)",
    (212, 1): "General Note:Dual Stack",
    (212, 2): "General Note:Imbedded Font Change",
    (212, 3): "General Note:Superscript",
    (212, 4): "General Note:Subscript",
    (212, 5): "General Note:Superscript, Subscript",
    (212, 6): "General Note:Multiple Stack, Left Justify",
    (212, 7): "General Note:Multiple Stack, Center Justify",
    (212, 8): "General Note:Multiple Stack, Right Justify",
    (212, 100): "General Note:Simple Fraction",
    (212, 101): "General Note:Dual Stack Fraction",
    (212, 102): "General Note:Imbedded Font Change, Double Fraction",
    (212, 105): "General Note:Superscript, Subscript Fraction",
    (214, 1): "Leader/Arrow:Wedge",
    (214, 2): "Leader/Arrow:Triangle",
    (214, 3): "Leader/Arrow:Filled Triangle",
    (214, 4): "Leader/Arrow:No Arrowhead",
    (214, 5): "Leader/Arrow:Circle",
    (214, 6): "Leader/Arrow:Filled Circle",
    (214, 7): "Leader/Arrow:Rectangle",
    (214, 8): "Leader/Arrow:Filled Rectangle",
    (214, 9): "Leader/Arrow:Slash",
    (214, 10): "Leader/Arrow:Integral Sign",
    (214, 11): "Leader/Arrow:Open Triangle",
    (216, 0): "Linear Dimension:Undetermined (default)",
    (216, 0): "Ordinate Dimension:Witness or Leader",
    (220, 0): "Point Dimension",
    (222, 0): "Radius Dimension:Single Leader Format",
    (228, 0): "General Symbol:General Symbol (default)",
    (228, 1): "General Symbol:Datum Feature Symbol",
    (228, 2): "General Symbol:Datum Target Symbol",
    (228, 3): "General Symbol:Feature Control Symbol",
    (230, 0): "Sectioned Area:Default",
    (308, 0): "Subfigure Definition",
    (404, 0): "Drawing:Normal Drawing (default)",
    (406, 0): Property,
    (406, 15): "Property:Name",
    (406, 16): "Property:Drawing Size",
    (406, 17): "Property:Drawing Units",
    (408, 0): "Singular Subfigure Instance",
    (410, 0): "View:Orthogonal (default)",
}


# -----------------------------------------------------------------------------
class IGESfile(object):
    _re_sep = re.compile("^1H(?P<sep_par>.).1H(?P<sep_rec>.).*$")
    _re_str = re.compile("[0-9]+H(?P<str_val>.*)")
    _re_flt = re.compile("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
    _re_stk = re.compile("(?P<first>[ ]*[0-9+-eE]+[^Ee])-[0-9+-eE]")
    _re_sqn = re.compile(".*[ ,]+(?P<seq_num>[0-9]+)\Z")
    (START, GLOBAL, DIRECTORY, PARAMETER, TERMINATE) = range(5)
    _tb_sec = {}
    _tb_ent = ENTTABLE

    def __init__(self, filename=None):
        self.de_counter = 0
        self.sep_par = ","
        self.sep_rec = ";"
        self.lineslist = []
        self.comment = []
        self.directory = []
        IGESfile._tb_sec = {
            "S": (IGESfile.START, "Start"),
            "G": (IGESfile.GLOBAL, "Global"),
            "D": (IGESfile.DIRECTORY, "Directory"),
            "P": (IGESfile.PARAMETER, "Parameter"),
            "T": (IGESfile.TERMINATE, "Terminate"),
        }
        if filename is not None:
            fd = open(sys.argv[1])
            self.lineslist = fd.readlines()
            fd.close()
            self.detect()

    def entries(self, code=None):
        ecount = 1
        pcount = 1
        for e in self.directory:
            if code in [None, e[0]]:
                yield ((ecount, pcount), e)
            ecount += 1
            pcount += 2

    def dump(self):
        print(self.comment)
        for de in self.directory:
            print(de)

    def parse(self):
        self.comment = self.snapcomment(self.snaptag(IGESfile.GLOBAL))
        self.directory = self.snapdirectory(self.snaptag(IGESfile.DIRECTORY))
        params = []
        self.de_counter = 1
        for l in self.snaprecord(self.snapsection(IGESfile.PARAMETER))[:-1]:
            params.append(self.snapparameter(l))
            IGESentity._parameters[self.de_counter] = self
            self.de_counter += 2
        for en in range(len(self.directory)):
            self.directory[en].append(params[en][1:])

    def snaptag(self, tag):
        return [l[:72] for l in self.lineslist if IGESfile._tb_sec[l[72]][0] == tag]

    def snapsection(self, tag):
        s = self.snaptag(tag)
        rl = []
        for sl in s:
            g = IGESfile._re_sqn.match(sl)
            try:
                seq_num = g.group("seq_num")
                rl += [sl[: -len(seq_num)]]
            except (AttributeError,):
                rl += sl
        return "".join(rl)

    def snaprecord(self, line):
        return line.split(self.sep_rec)

    def snapparameter(self, line):
        tkl = [self.snapsticky(tk) for tk in line.split(self.sep_par)]
        return [self.guesstype(tk) for tk in tkl]

    def snapcomment(self, line):
        return line

    def snapdirectory(self, line):
        tkl = []
        two = False
        for de in line:
            self.de_counter += 1
            tk = [
                self.guesstype(t)
                for t in [
                    de[:9],
                    de[8:16],
                    de[16:24],
                    de[24:32],
                    de[32:40],
                    de[40:48],
                    de[48:56],
                    de[56:64],
                    de[64:72],
                ]
            ]
            if two:
                two = False
                tkp += tk[1:]
                tkl.append(tkp)
            else:
                two = True
                tkp = tk
        return tkl

    def snapsticky(self, token):
        g = IGESfile._re_stk.match(token)
        try:
            first = g.group("first")
            token = first.strip() + self.sep_par + token[len(first) :]
        except (AttributeError,):
            pass
        return token

    def detect(self):
        for l in self.snapsection(IGESfile.GLOBAL):
            if self.sepdetect(l):
                break

    def sepdetect(self, line):
        g = IGESfile._re_sep.match(line)
        try:
            self.sep_par = g.group("sep_par")
            self.sep_rec = g.group("sep_rec")
            return True
        except (AttributeError,):
            return False

    def guesstype(self, token):
        try:
            v = int(token)
            return v
        except ValueError:
            pass
        try:
            v = float(token)
            return v
        except ValueError:
            pass
        g = IGESfile._re_str.match(token)
        try:
            v = g.group("str_val")
            return v
        except (AttributeError,):
            pass
        if token == "        ":
            token = None
        return token


# -----------------------------------------------------------------------------
def IGESfactory(index, direntry):
    code = (direntry[0], 0)
    if code in ENTTABLE and not isinstance(ENTTABLE[code], str):
        return ENTTABLE[code](index, direntry[1:])
    return IGESentity(direntry[0], "", index, direntry[1:])


# -----------------------------------------------------------------------------
class IGESbase:
    _cnt = 1

    def __init__(self, name):
        self.name = name
        self.entities = []

    def addentity(self, ent):
        self.entities.append(ent)

    def bind(self):
        IGESentity.solve()

    def dump(self):
        for e in IGESentity.sortedentities():
            print(e)

    def nodeps(self):
        a = {x.index[0] for x in IGESentity.sortedentities()}
        for e in IGESentity.sortedentities():
            b = {x for x in e.allchildren()}
            a.difference_update(b)
        r = list(a)
        r.sort()
        return r

    def families(self):
        f = []
        for e in IGESentity.allentities(406):
            f += [(e.property, e.index[0])]
        return f

    def parsechildren(self, index, name):
        e = IGESentity.find(index)
        node = [name, None, [], "UserDefinedData_t"]
        node[2] += e.asCGNS()
        if e.code == 406:
            fname = e.property
        else:
            fname = None
        for c in e.allchildren():
            if c not in e.childrenname:
                name = "Entity#%.4d" % c
            else:
                name = e.childrenname[c]
            f, n = self.parsechildren(c, name)
            node[2].append(n)
            if f:
                fname = f
        return (fname, node)

    def tree(self, topindex):
        (f, r) = self.parsechildren(topindex, "Entity#%.4d" % topindex)
        if f is None:  # check if there's a label
            f = IGESentity.find(topindex).label()
        if f in ["", None]:
            f = "Unknown#%.4d" % IGESbase._cnt
            IGESbase._cnt += 1
        return f, r


# -----------------------------------------------------------------------------
def main():
    args = parse()
    debug = 1
    if len(args.files) < 2:
        sys.exit(1)

    f = IGESfile(args.files[0])
    f.parse()
    db = IGESbase(args.files[0])

    for ix, de in f.entries():
        e = IGESfactory(ix, de)
        db.addentity(e)

    db.bind()
    db.dump()

    basename = "B"

    T = CGL.newCGNSTree()
    b = CGL.newCGNSBase(T, basename, 3, 3)

    toplist = db.nodeps()

    for top in toplist:
        (fname, et) = db.tree(top)
        f = CGL.newFamily(b, fname)
        g = CGL.newGeometryReference(f, "Geometry IGES", CGK.NASAIGES_s)
        CGU.setAsChild(g, et)

    if debug:
        for p in CGU.getPathsFullTree(T):
            print(p)
        print(CGU.prettyPrint(T))

    CGM.save(args.files[1], T)


if __name__ == "__main__":
    main()


# --- last line
