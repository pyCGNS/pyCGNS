#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnserrors as E
import CGNS.PAT.cgnskeywords as K
import numpy as N

# RegionCellDimension
data = [
    "{ZoneSubRegion}",
    N.array([3], dtype=N.int32, order="F"),
    [],
    K.ZoneSubRegion_ts,
]

C.newGridLocation(data, K.Vertex_s)
C.newPointRange(data)
C.newFamilyName(data, "{Family}")
C.newUserDefinedData(data, "{UserDefinedData}")
C.newDescriptor(data, "{Descriptor}")

status = "7.3"
comment = "partial SIDS with some optionals"
pattern = [data, status, comment]
