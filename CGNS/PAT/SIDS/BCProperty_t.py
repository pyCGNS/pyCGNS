#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnserrors as E
import CGNS.PAT.cgnskeywords as K
import numpy as N

#
data = C.newBCProperty(None)
C.newDescriptor(data, "{Descriptor}")
C.newUserDefinedData(data, "{UserDefinedData}")
# for n in data[2]:
#    if n[0] == K.Area_s:
#        C.newArea(n)
#    if (n[0] == K.WallFunction_s):
#        C.newDescriptor(n, '{Descriptor}')
#        C.newUserDefinedData(n, '{UserDefinedData}')
# #
status = "9.6"
comment = "Full SIDS with all optionals"
pattern = [data, status, comment]
#
