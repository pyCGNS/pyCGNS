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

UKZONETYPE='S004'
BADVALUEDTYPE='S100'
BADVALUESHAPE='S101'
BADCELLDIM='S102'
BADPHYSDIM='S103'

class SIDSbase(CGNS.VAL.parse.generic.GenericParser):
  __messages={
   UKZONETYPE:'Unknown ZoneType_t value',
   BADVALUEDTYPE:'Bad node value data type',
   BADVALUESHAPE:'Bad node value shape',
   BADCELLDIM:'Bad value for CellDimensions',
   BADPHYSDIM:'Bad value for PhysicalDimensions',
  }
  def __init__(self,log):
    CGNS.VAL.parse.generic.GenericParser.__init__(self,log)
    self.log.addMessages(SIDSbase.__messages)
  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,CGM.CHECK_FAIL,BADVALUESHAPE)
    else:
      cd=node[1][0]
      pd=node[1][1]
      self.context[CGK.CellDimension_s]=0
      self.context[CGK.PhysicalDimension_s]=0
      if (cd not in [1,2,3]):
        rs=log.push(pth,CGM.CHECK_FAIL,BADCELLDIM)
        self.context[CGK.CellDimension_s]=cd
      if (pd not in [1,2,3]):
        rs=log.push(pth,CGM.CHECK_FAIL,BADPHYSDIM)
        self.context[CGK.PhysicalDimension_s]=pd
    return rs
  # --------------------------------------------------------------------
  def ZoneType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.ZoneType_l)):
      rs=log.push(pth,CGM.CHECK_FAIL,UKZONETYPE)
    return rs
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
