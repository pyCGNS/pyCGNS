#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
#
import CGNS.VAL.simplecheck as CGV
import CGNS.MAP
import importlib
import sys
import os
import string

KEY = "SAMPLE"
FILES = False
CGNSCHECK = False
TRACE = False

try:
    import CGNS.PRO

    KEY = "SIDS"
except ImportError:
    pass

try:
    if "CGNS_VAL_SAVE_FILES" in os.environ:
        FILES = True
except:
    pass

try:
    if "CGNS_VAL_RUN_CGNSCHECK" in os.environ:
        CGNSCHECK = True
except:
    pass


def cgcheck(filenametag):
    os.system("cgnscheck %s.hdf 1>%s.cgnscheck 2>&1" % (filenametag, filenametag))


def runSuite(suite, trace, savefile=False, cgnscheck=False):
    s = importlib.import_module(".%s" % suite, "CGNS.VAL.suite")
    tlist = []
    count = 1
    if s is not None:
        p = os.path.split(s.__file__)[0]
        l = os.listdir(p)
        for t in l:
            if (
                (t[0] != "_")
                and (t[-3:] == ".py")
                and (t[0] in string.digits)
                and (t[1] in string.digits)
            ):
                tlist.append(t[:-3])
    for t in tlist:
        tdlist = loadTree(suite, t)
        for (tag, T, diag) in tdlist:
            sr = valTree(suite, t, tag, T, diag, trace, count)
            if diag:
                k = "pass"
            else:
                k = "fail"
            if savefile:
                fname = "%.4d-%s-%s-%s.hdf" % (count, k, suite, t)
                try:
                    CGNS.MAP.save(fname, T)
                except CGNS.MAP.error as e:
                    sr += "### * CHLone save error %d:%s\n" % (e[0], e[1])
                f = open("%.4d-%s-%s-%s.diag" % (count, k, suite, t), "w+")
                f.write(sr)
                f.close()
                if cgnscheck:
                    cgcheck("%.4d-%s-%s-%s" % (count, k, suite, t))
            count += 1


def loadTree(key, test):
    m = importlib.import_module(".%s" % test, "CGNS.VAL.suite.%s" % key)
    return m.TESTS


def valTree(suite, t, tag, T, diag, trace, count):
    sr = ""
    r = CGV.compliant(T, False, [KEY])
    if diag:
        k = "pass"
    else:
        k = "fail"
    if r[0] == diag:
        (v1, v2) = ("pass", "expected")
    else:
        (v1, v2) = ("FAIL", "UNEXPECTED")
    if trace:
        sr += "###\n"
    sr += "### [%s/%s] %.4d-%s-%s-%s (%s)\n" % (KEY, v1, count, k, suite, t, tag)
    for m in r[1]:
        sr += "### > %s error on: %s\n" % (v2, m[0])
        sr += "### > %s\n" % (m[1])
    if trace:
        print(sr)
    return sr


def runall():
    tlist = ("SIDS",)
    for s in tlist:
        KEY = s
        runSuite(s, TRACE, FILES, CGNSCHECK)
