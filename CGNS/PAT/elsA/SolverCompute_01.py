#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnsutils    as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

from .SolverCompute import data

CGL.newDataArray(data,'turbmod',CGU.setStringAsArray("rsm"))
CGL.newDataArray(data,'rsm_name',CGU.setStringAsArray("ssg_chien"))
CGL.newDataArray(data,'t_cutvar',CGU.setDoubleAsArray(1.78929762604e-08))
CGL.newDataArray(data,'t_harten',CGU.setDoubleAsArray(1e-24))

status='0.1'
comment='Parameter set for RSM'
pattern=[data, status, comment]
