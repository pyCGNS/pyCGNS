# -------------------------------------------------------------------------
# pyCGNS - Python package for CFD General Notation System -
# See license.txt file in the root directory of this Python module source  
# -------------------------------------------------------------------------
#
# this script doesn't need to be used for every doc update, run it once
# using :
#   python lib/gendoc.py > doc/mods/PAT/cgnstypes.txt
#
#
import CGNS.PAT.cgnstypes

nzero = [CGNS.PAT.cgnstypes.C_11, CGNS.PAT.cgnstypes.C_1N, CGNS.PAT.cgnstypes.C_NN]


def gentype2(t):
    s = "\n-----\n\n.. _X%s:\n\n%s\n%s\n" % (t.type, t.type, len(t.type) * '-')
    for nt in t.names:
        if (nt == '{UserDefined}'):
            s += "\n :Name: %s " % nt
        else:
            s += "\n :Name: **%s** " % nt
    for c in t.parents:
        s += "\n :Parent: :ref:`%s <X%s>` " % (c, c)
    dtf = False
    for dt in t.datatype:
        if dt != 'LK':
            if not dtf:
                s += "\n :DataType: "
                dtf = True
            s += " " + dt
    if t.enumerate:
        s += "\n :Enumerate: "
    for c in t.children:
        for cn in c[1]:
            cc = t.cardinality(c[0])
            if ((cn != '{UserDefined}') and (t.cardinality(c[0]) not in nzero)):
                s += "\n :Child: **%s** :ref:`%s <X%s>` (%s)" % (cn, c[0], c[0], cc)
            else:
                s += "\n :Child: %s :ref:`%s <X%s>` (%s)" % (cn, c[0], c[0], cc)
    s += "\n"
    return s


def gentype(t):
    s = "\n-----\n\n.. _X%s:\n\n%s\n%s\n" % (t.type, t.type, len(t.type) * '-')
    s += "\n * Name: "
    for nt in t.names:
        s += "\n\n   - %s " % nt
    s += "\n\n * Data-type: "
    for dt in t.datatype:
        if dt != 'LK':
            s += " " + dt
    if t.enumerate:
        s += "\n * Enumerate: "
    s += "\n * Cardinality: %s" % t.cardinality
    s += "\n * Children\n"
    for c in t.children:
        s += "\n   - :ref:`%s <X%s>` (%s)" % (c[0], c[0], c[1])
    s += "\n\n * Parents\n"
    for c in t.parents:
        s += "\n   - :ref:`%s <X%s>` " % (c, c)
    s += "\n"
    return s


def gentypes():
    s = ""
    ct = CGNS.PAT.cgnstypes.types.keys()
    ct.sort()
    for c in ct:
        s += gentype2(CGNS.PAT.cgnstypes.types[c])
    return s


print(gentypes())

# --- last line
