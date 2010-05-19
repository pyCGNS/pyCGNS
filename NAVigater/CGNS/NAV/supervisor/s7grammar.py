#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.NAV.gui.s7globals
G___=CGNS.NAV.gui.s7globals.s7G

import s7parser
try:
  import CGNS.PAT.cgnskeywords as CK
except ImportError:
  pass
#  s7utils.importCGNSWarning('SIDS pattern')

# --------------------------------------------------------------------
# Enumerates
def ZoneType_t_enum():
  return CK.ZoneType_l

def SimulationType_t_enum():
  return CK.SimulationType_l

def RigidGridMotionType_t_enum():
  return CK.RigidGridMotionType_l

def TurbulenceModelType_t_enum():
  return CK.TurbulenceModelType_l

def ViscosityModelType_t_enum():
  return CK.ViscosityModelType_l

def TurbulenceClosureType_t_enum():
  return CK.TurbulenceClosureType_l

def GasModelType_t_enum():
  return CK.GasModelType_l

def ThermalRelaxationModelType_t_enum():
  return CK.ThermalRelaxationModelType_l

def ChemicalKineticsModelType_t_enum():
  return CK.ChemicalKineticsModelType_l

def EMElectricFieldModelType_t_enum():
  return CK.EMElectricFieldModelType_l

def EMMagneticFieldModelType_t_enum():
  return CK.EMMagneticFieldModelType_l

def EMConductivityModelType_t_enum():
  return CK.EMConductivityModelType_l

def AverageInterfaceType_t_enum():
  return CK.AverageInterfaceType_l

def GoverningEquationsType_t_enum():
  return CK.GoverningEquationsType_l

def ElementType_t_enum():
  return CK.ElementType_l

def ArbitraryGridMotionType_t_enum():
  return CK.ArbitraryGridMotionType_l

# --- special cases
def BCTypeSimple_t_enum():
  return CK.BCTypeSimple_l

def BCTypeCompound_t_enum():
  return CK.BCTypeCompound_l

def MassUnits_t_enum():
  return CK.MassUnits_l

def LengthUnits_t_enum():
  return CK.LengthUnits_l

def TimeUnits_t_enum():
  return CK.TimeUnits_l

def TemperatureUnits_t_enum():
  return CK.TemperatureUnits_l

def AngleUnits_t_enum():
  return CK.AngleUnits_l

def LuminousIntensityUnits_t_enum():
  return CK.LuminousIntensityUnits_l

def DataClass_t_enum():
  return CK.DataClass_l

def GridLocation_t_enum():
  return CK.GridLocation_l

def GridConnectivityType_t_enum():
  return CK.GridConnectivityType_l

# --------------------------------------------------------------------
def Zone_t(pth,node,parent,tree,check,log):
  rs=1
  r=0
  msg='## No GridCoordinates in this Zone_t'
  for cn in s7parser.childNames(node):
    if (cn!='GridCoordinates'):
      r=1
      break
  if (not r): log.push("\n%s\n"%(msg),'#FAIL')
  rs*=r
  r=0
  msg='## No FlowSolution# found for output definition'
  for cn in s7parser.childNames(node):
    if (    (len(cn)>12)
        and (cn[:13]=='FlowSolution#')
        and (cn!='FlowSolution#Init')):
      r=1
      break
  if (not r): log.push("\n%s\n"%(msg),'#WARNING')
  rs*=r
  r=0
  msg='## No FlowSolution#Init found for field initialization'
  for cn in s7parser.childNames(node):
    if (cn=='FlowSolution#Init'):
      r=1
      break
  if (not r): log.push("\n%s\n"%(msg),'#WARNING')
  rs*=r
  return rs

# --------------------------------------------------------------------
def CGNSBase_t(pth,node,parent,tree,check,log):
  rs=1
  r=0
  msg='## No Zone_t found in this CGNSBase_t'
  r=s7parser.hasChildNodeOfType(node,'Zone_t')
  if (not r): log.push("\n%s\n"%(msg),'#FAIL')
  rs*=r
  r=0
  msg='## No ReferenceState found in this CGNSBase_t'
  for cn in s7parser.childNames(node):
    if (cn!='ReferenceState'):
      r=1
      break
  if (not r): log.push("\n%s\n"%(msg),'#WARNING')
  rs*=r
  return rs

# --------------------------------------------------------------------
def ZoneType_t(pth,node,parent,tree,check,log):
  msg='## Only Structured ZoneType_t is allowed'
  r=s7parser.stringValueMatches(node,'Structured')
  if (not r): log.push("\n%s\n"%(msg),'#FAIL')
  return 1

# --------------------------------------------------------------------
def IndexRange_t(pth,node,parent,tree,check,log):
  if not ((node[0]==CK.PointRange_s) or (node[0]==CK.PointRangeDonor_s)):
    return 1
  if (node[2]):
    return 0
  if not ((len(node[1])==2) and (len(node[1][0]==3)) and (len(node[1][1]==3))):
    return 0
  return 1

# --------------------------------------------------------------------
def IndexRangeT2_t(pth,node,parent,tree,check,log):
  if not (node[0]==CK.Transform_s):
    return 1
  if (node[2]):
    return 0
  if not (len(node[1])==3):
    return 0
  for n in node[1]:
    if (n not in [1,2,3,-1,-2,-3]):
      return 0
  return 1
