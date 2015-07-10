#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnsutils    as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=CGL.newUserDefinedData(None,'Solver#Compute')
CGL.newDataArray(data,'artviscosity',CGU.setStringAsArray("dissca"))
CGL.newDataArray(data,'avcoef_k2',CGU.setDoubleAsArray(1.0))
CGL.newDataArray(data,'avcoef_k4',CGU.setDoubleAsArray(0.032))
CGL.newDataArray(data,'avcoef_sigma',CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'fluid',CGU.setStringAsArray("Ideal"))
CGL.newDataArray(data,'flux',CGU.setStringAsArray("jameson"))
CGL.newDataArray(data,'ode',CGU.setStringAsArray("rk4"))
CGL.newDataArray(data,'phymod',CGU.setStringAsArray("NSTurbulent"))
CGL.newDataArray(data,'time_algo',CGU.setStringAsArray("steady"))
CGL.newDataArray(data,'visclaw',CGU.setStringAsArray("Sutherland"))
CGL.newDataArray(data,'walldistcompute',CGU.setStringAsArray("mininterf_ortho"))

CGL.newDataArray(data,'niter',CGU.setIntegerAsArray(1000))

status='0.1'
comment='Set of usual parameters for elsA computation'
pattern=[data, status, comment]
