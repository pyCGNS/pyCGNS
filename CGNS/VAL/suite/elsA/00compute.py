#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy as NPY

TESTS=[]

#  -------------------------------------------------------------------------
tag='complete compute'
diag=True
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base#1}',3,3)
c=CGL.newUserDefinedData(b,'.Solver#Compute')
d=CGL.newDataArray(c,'artviscosity',CGU.setStringAsArray("dissca"))
d=CGL.newDataArray(c,'avcoef_k2',CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(c,'niter',CGU.setIntegerAsArray(1000))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='missing compute node'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base#1}',3,3)
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
