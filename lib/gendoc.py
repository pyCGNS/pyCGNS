#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#
import CGNS.PAT.cgnstypes

def gentype(t):
    s="\n-----\n\n.. _X%s:\n\n%s\n%s\n"%(t.type,t.type,len(t.type)*'-')
    s+="\n * Name: "
    for nt in t.names:
      s+="\n\n   - %s "%nt
    s+="\n\n * Data-type: "
    for dt in t.datatype:
        if (dt != 'LK'): s+=" "+dt
    if (t.enumerate):
        s+="\n * Enumerate: "
    s+="\n * Cardinality: %s"%t.cardinality
    s+="\n * Children\n"
    for c in t.children:
        s+="\n   - :ref:`%s <X%s>` (%s)"%(c[0],c[0],c[1])
    s+="\n\n * Parents\n"
    for c in t.parents:
        s+="\n   - :ref:`%s <X%s>` "%(c,c)
    s+="\n"
    return s
        
def gentypes():
    s=""
    ct=CGNS.PAT.cgnstypes.types.keys()
    ct.sort()
    for c in ct:
        s+=gentype(CGNS.PAT.cgnstypes.types[c])
    return s

print gentypes()

# --- last line
