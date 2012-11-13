#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils          as CGU
import CGNS.PAT.cgnstypes          as CGT
import CGNS.PAT.cgnskeywords       as CGK
import CGNS.VAL.parse.messages     as CGM
import CGNS.VAL.parse.generic

messagetable=(
('NOZONE','U101',CGM.CHECK_WARN,'No Zone in this Base'),
('NOGRIDZONE','U102',CGM.CHECK_WARN,'No GridCoordinates in this Zone'),
('NOSTRUCTZONE','U105',CGM.CHECK_FAIL,'At least one structured Zone is required in the Base'),
('NOBREFSTATE','U103',CGM.CHECK_WARN,'No ReferenceState found at Base level'),
('NOZREFSTATE','U104',CGM.CHECK_WARN,'No ReferenceState found at Zone level'),
('NOFLOWSOL','U107',CGM.CHECK_WARN,'No FlowSolution# found for output definition'),
('NOFLOWINIT','U108',CGM.CHECK_WARN,'No FlowSolution#Init found for fields initialisation'),
('NOTRHTRANSFORM','U106',CGM.CHECK_FAIL,'Transform is not right-handed (direct)'),
('CHSGRIDLOCATION','U109',CGM.CHECK_FAIL,'Cannot handle such GridLocation [%s]'),
('CHSELEMENTTYPE','U110',CGM.CHECK_FAIL,'Cannot handle such ElementType [%s]'),
)

USER_MESSAGES={}
for (v,k,l,m) in messagetable:
  locals()[v]=k
  USER_MESSAGES[k]=(l,m)

# -----------------------------------------------------------------------------
class CGNS_VAL_USER_Checks(CGNS.VAL.parse.generic.GenericParser):
  def __init__(self,log):
    CGNS.VAL.parse.generic.GenericParser.__init__(self,log)
    self.log.addMessages(USER_MESSAGES)
  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGK.GridCoordinates_s not in CGU.childNames(node)):
      rs=log.push(pth,NOGRIDZONE)
    if (not CGU.hasChildNodeOfType(node,CGK.ReferenceState_ts)):
      rs=log.push(pth,NOZREFSTATE)
    return rs
  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.hasChildNodeOfType(node,CGK.Zone_ts)):
      rs=log.push(pth,NOZONE)
    else:
      target=[CGK.CGNSTree_ts,node[0],CGK.Zone_ts,CGK.ZoneType_s]
      plist=CGU.getAllNodesByTypeOrNameList(tree,target)
      found=False
      for p in plist:
        if (CGU.stringValueMatches(CGU.getNodeByPath(tree,p),
                                   CGK.Structured_s)):
          found=True
      if (not found):
        rs=log.push(pth,NOSTRUCTZONE)
    if (not CGU.hasChildNodeOfType(node,CGK.ReferenceState_ts)):
      rs=log.push(pth,NOBREFSTATE)
    return rs
  # --------------------------------------------------------------------
  def GridLocation_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    val=node[1].tostring()
    if (val not in [CGK.Vertex_s,CGK.CellCenter_s,CGK.FaceCenter_s]):
        rs=log.push(pth,CHSGRIDLOCATION,val)
    return rs
# -----
