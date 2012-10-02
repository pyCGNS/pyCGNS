#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils      as CGU
import CGNS.PAT.cgnstypes      as CGT
import CGNS.PAT.cgnskeywords   as CGK
import CGNS.VAL.parse.messages as CGM
import CGNS.VAL.parse.generic

import string

class SIDSbase(CGNS.VAL.parse.generic.GenericParser):
  def __init__(self):
    CGNS.VAL.parse.generic.GenericParser.__init__(self)
    self.updateMethods()
    
  # --------------------------------------------------------------------
  # Enumerates
  def ZoneType_t_enum(self):
    return CGK.ZoneType_l

  def SimulationType_t_enum(self):
    return CGK.SimulationType_l

  def RigidGridMotionType_t_enum(self):
    return CGK.RigidGridMotionType_l

  def TurbulenceModelType_t_enum(self):
    return CGK.TurbulenceModelType_l

  def ViscosityModelType_t_enum(self):
    return CGK.ViscosityModelType_l

  def TurbulenceClosureType_t_enum(self):
    return CGK.TurbulenceClosureType_l

  def GasModelType_t_enum(self):
    return CGK.GasModelType_l

  def ThermalRelaxationModelType_t_enum(self):
    return CGK.ThermalRelaxationModelType_l

  def ChemicalKineticsModelType_t_enum(self):
    return CGK.ChemicalKineticsModelType_l

  def EMElectricFieldModelType_t_enum(self):
    return CGK.EMElectricFieldModelType_l

  def EMMagneticFieldModelType_t_enum(self):
    return CGK.EMMagneticFieldModelType_l

  def EMConductivityModelType_t_enum(self):
    return CGK.EMConductivityModelType_l

  def AverageInterfaceType_t_enum(self):
    return CGK.AverageInterfaceType_l

  def GoverningEquationsType_t_enum(self):
    return CGK.GoverningEquationsType_l

  def ElementType_t_enum(self):
    return CGK.ElementType_l

  def ArbitraryGridMotionType_t_enum(self):
    return CGK.ArbitraryGridMotionType_l

  # --- special cases
  def BCTypeSimple_t_enum(self):
    return CGK.BCTypeSimple_l

  def BCTypeCompound_t_enum(self):
    return CGK.BCTypeCompound_l

  def MassUnits_t_enum(self):
    return CGK.MassUnits_l

  def LengthUnits_t_enum(self):
    return CGK.LengthUnits_l

  def TimeUnits_t_enum(self):
    return CGK.TimeUnits_l

  def TemperatureUnits_t_enum(self):
    return CGK.TemperatureUnits_l

  def AngleUnits_t_enum(self):
    return CGK.AngleUnits_l

  def LuminousIntensityUnits_t_enum(self):
    return CGK.LuminousIntensityUnits_l

  def DataClass_t_enum(self):
    return CGK.DataClass_l

  def GridLocation_t_enum(self):
    return CGK.GridLocation_l

  def GridConnectivityType_t_enum(self):
    return CGK.GridConnectivityType_l

  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,log):
    rs=1
    r=0
    msg='No GridCoordinates in this Zone_t'
    for cn in CGU.childNames(node):
      if (cn!='GridCoordinates'):
        r=1
        break
    if (not r): log.push(pth,CGM.CHECK_WARN,msg)
    rs*=r
    r=0
    msg='No FlowSolution# found for output definition'
    for cn in CGU.childNames(node):
      if (    (len(cn)>12)
          and (cn[:13]=='FlowSolution#')
          and (cn!='FlowSolution#Init')):
        r=1
        break
    if (not r): log.push(pth,CGM.CHECK_WARN,msg)
    rs*=r
    r=0
    msg='No FlowSolution#Init found for field initialization'
    for cn in CGU.childNames(node):
      if (cn=='FlowSolution#Init'):
        r=1
        break
    if (not r): log.push(pth,CGM.CHECK_WARN,msg)
    rs*=r
    return rs

  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    rs=1
    r=0
    msg='No Zone_t found in this CGNSBase_t'
    r=CGU.hasChildNodeOfType(node,CGK.Zone_ts)
    if (not r): log.push(pth,CGM.CHECK_WARN,msg)
    rs*=r
    r=0
    msg='No ReferenceState found in this CGNSBase_t'
    for cn in CGU.childNames(node):
      if (cn!=CGK.ReferenceState_s):
        r=1
        break
    if (not r): log.push(pth,CGM.CHECK_WARN,msg)
    rs*=r
    return rs

  # --------------------------------------------------------------------
  def ZoneType_t(self,pth,node,parent,tree,log):
    msg='Only Structured ZoneType_t is allowed'
    r=CGU.stringValueMatches(node,CGK.Structured_s)
    if (not r): log.push(pth,CGM.CHECK_WARN,msg)
    return 1

  # --------------------------------------------------------------------
  def IndexRange_t(self,pth,node,parent,tree,log):
    if not ((node[0]==CGK.PointRange_s) or (node[0]==CGK.PointRangeDonor_s)):
      return 1
    if (node[2]):
      return 0
    if not ((len(node[1])==2) and (len(node[1][0]==3)) and (len(node[1][1]==3))):
      return 0
    return 1

  # --------------------------------------------------------------------
  def IndexRangeT2_t(self,pth,node,parent,tree,log):
    if not (node[0]==CGK.Transform_s):
      return 1
    if (node[2]):
      return 0
    if not (len(node[1])==3):
      return 0
    for n in node[1]:
      if (n not in [1,2,3,-1,-2,-3]):
        return 0
    return 1

# -----
