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
import CGNS.VAL.grammars.SIDS

import string

NOZONE            = 'U101'
NOGRIDZONE        = 'U102'
NOBREFSTATE       = 'U103'
NOZREFSTATE       = 'U104'
NOSTRUCTZONE      = 'U105'
NOTRHTRANSFORM    = 'U106'
NOFLOWSOL         = 'U107'
NOFLOWINIT        = 'U108'
CHSGRIDLOCATION   = 'U109'
CHSELEMENTTYPE    = 'U110'

class elsAbase(CGNS.VAL.grammars.SIDS.SIDSbase):
  __messages={
NOZONE:'No Zone in this Base',
NOGRIDZONE:'No GridCoordinates in this Zone',
NOSTRUCTZONE:'At least one structured Zone is required in the Base',
NOBREFSTATE:'No ReferenceState found at Base level',
NOZREFSTATE:'No ReferenceState found at Zone level',
NOFLOWSOL:'No FlowSolution# found for output definition',
NOFLOWINIT:'No FlowSolution#Init found for fields initialisation',
NOTRHTRANSFORM:'Transform is not right-handed (direct)',
CHSGRIDLOCATION:'Cannot handle such GridLocation [%s]',
CHSELEMENTTYPE:'Cannot handle such ElementType [%s]',
  }
  def __init__(self,log):
    CGNS.VAL.grammars.SIDS.SIDSbase.__init__(self,log)
    self.log.addMessages(elsAbase.__messages)
    self.sids=CGNS.VAL.grammars.SIDS.SIDSbase
  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,log):
    rs=self.sids.Zone_t(self,pth,node,parent,tree,log)
    if (CGK.GridCoordinates_s not in CGU.childNames(node)):
      rs=log.push(pth,CGM.CHECK_WARN,NOGRIDZONE)
    if (not CGU.hasChildNodeOfType(node,CGK.ReferenceState_ts)):
      rs=log.push(pth,CGM.CHECK_WARN,NOZREFSTATE)
    return rs
  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    rs=self.sids.CGNSBase_t(self,pth,node,parent,tree,log)
    if (not CGU.hasChildNodeOfType(node,CGK.Zone_ts)):
      rs=log.push(pth,CGM.CHECK_WARN,NOZONE)
    else:
      target=[CGK.CGNSTree_ts,node[0],CGK.Zone_ts,CGK.ZoneType_s]
      plist=CGU.getAllNodesByTypeOrNameList(tree,target)
      found=0
      for p in plist:
        if (CGU.stringValueMatches(CGU.getNodeByPath(tree,p),
                                   CGK.Structured_s)):
          found=0
      if (not found):
        rs=log.push(pth,CGM.CHECK_FAIL,NOSTRUCTZONE)
    if (not CGU.hasChildNodeOfType(node,CGK.ReferenceState_ts)):
      rs=log.push(pth,CGM.CHECK_WARN,NOBREFSTATE)
    return rs
  # --------------------------------------------------------------------
  def IntIndexDimension_t(self,pth,node,parent,tree,log):
    rs=self.sids.IntIndexDimension_t(self,pth,node,parent,tree,log)
    if (node[0]==CGK.Transform_s):
      tr=list(node[0].flat)
      if (not CGS.transformIsDirect(tr,self.context[CGK.CellDimension_s])):
        rs=log.push(pth,CGM.CHECK_FAIL,NOTRHTRANSFORM)
    return rs
  # --------------------------------------------------------------------
  def GridLocation_t(self,pth,node,parent,tree,log):
    rs=self.sids.GridLocation_t(self,pth,node,parent,tree,log)
    val=node[1].tostring()
    if (val not in [CGK.Vertex_s,CGK.CellCenter_s,CGK.FaceCenter_s]):
        rs=log.push(pth,CGM.CHECK_FAIL,CHSGRIDLOCATION,val)
    return rs
  # --------------------------------------------------------------------
  def Elements_t(self,pth,node,parent,tree,log):
    rs=self.sids.Elements_t(self,pth,node,parent,tree,log)
    if (self.context[CGK.ElementType_s] not in [CGK.QUAD_4, CGK.HEXA_8]):
      try:
        et=CGK.ElementType_[self.context[CGK.ElementType_s]]
      except KeyError:
        et=str(self.context[CGK.ElementType_s])
      rs=log.push(pth,CGM.CHECK_FAIL,CHSELEMENTTYPE,et)
    return rs
# -----
