#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnskeywords as CK
import CGNS.PAT.cgnstypes as CT

stringsphinx0 = """

----

.. _X%s:

%s
%s

 * Name

   * %s
 """

stringsphinx1 = """

 * Data-Type: %s

 * Dimensions/DimensionValues

 * Cardinality: %s

"""

stringsphinx2 = """
 * Child Nodes

   - :ref:`%s <X%s>` (%s)\n"""

stringsphinx3 = """   - :ref:`%s <X%s>` (%s)\n"""

stringsphinx4 = """   * %s\n"""


# a ref is :ref:`XCGNSBase_t`
def showBoxes():
    s = ""
    lt = list(CT.types)
    lt.sort()
    for t in lt:
        st = "-" * len(t)
        s += stringsphinx0 % (t, t, st, CT.types[t].names[0])
        if len(CT.types[t].names) > 1:
            for n in CT.types[t].names[1:]:
                s += stringsphinx4 % n
        datatype = ""
        for d in CT.types[t].datatype:
            datatype += "%s " % d
        cardinality = CT.types[t].cardinality
        s += stringsphinx1 % (datatype, cardinality)
        if CT.types[t].children:
            s += stringsphinx2 % (
                CT.types[t].children[0][0],
                CT.types[t].children[0][0],
                CT.types[t].children[0][1],
            )
        if CT.types[t].children > 1:
            for c in CT.types[t].children[1:]:
                s += stringsphinx3 % (c[0], c[0], c[1])
    return s


def generateSphinx(file):
    f = open(file, "w+")
    f.write(showBoxes())
    f.close()


# --- last line
