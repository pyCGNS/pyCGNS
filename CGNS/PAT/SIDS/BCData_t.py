#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
from .. import cgnslib as C
from .. import cgnserrors as E
from .. import cgnskeywords as K
import numpy as N

#
data = C.newBCData(None, "{BCData}")
C.newDescriptor(data, "{Descriptor}")
C.newDataArray(data, "{DataLocal}")
C.newDataArray(data, "{DataGlobal}")
C.newDataClass(data)
C.newDimensionalUnits(data)
C.newUserDefinedData(data, "{UserDefinedData}")
#
status = "9.5"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
