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
CGL.newGridLocation(data,CGK.CellCenter_s)
CGL.newRind(data,NPY.array([0,0,0,0,0,0]))
CGL.newUserDefinedData(data,CGK.MomentumX_s)
CGL.newUserDefinedData(data,CGK.MomentumY_s)
CGL.newUserDefinedData(data,CGK.MomentumZ_s)
CGL.newUserDefinedData(data,CGK.Density_s)
CGL.newUserDefinedData(data,CGK.EnergyStagnationDensity_s)
CGL.newUserDefinedData(data,CGK.TurbulentDistance_s)
CGL.newUserDefinedData(data,"TurbulentEnergyKineticDensity")
CGL.newUserDefinedData(data,"TurbulentDissipationDensity")
CGL.newUserDefinedData(data,"TurbulentDistanceIndex")
CGL.newUserDefinedData(data,"TurbulentSANuTildeDensity")
    
status='0.1'
comment='FlowSolution pattern for end of run output'
pattern=[data, status, comment]
