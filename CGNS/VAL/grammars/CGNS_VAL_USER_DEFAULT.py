#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnstypes as CGT
import CGNS.PAT.cgnskeywords as CGK
import CGNS.VAL.parse.messages as CGM
import CGNS.VAL.parse.generic

messagetable = (
    ("U0101", CGM.CHECK_WARN, "No Zone in this Base"),
    ("U0102", CGM.CHECK_WARN, "No Structured Zone found"),
    ("U0105", CGM.CHECK_FAIL, "At least one structured Zone is required in the Base"),
    ("U0103", CGM.CHECK_WARN, "No ReferenceState found at Base level"),
    ("U0104", CGM.CHECK_WARN, "No ReferenceState found at Zone level"),
    ("U0107", CGM.CHECK_WARN, "No FlowSolution# found for output definition"),
    ("U0108", CGM.CHECK_WARN, "No FlowSolution#Init found for fields initialisation"),
    ("U0106", CGM.CHECK_FAIL, "Transform is not right-handed (direct)"),
    ("U0109", CGM.CHECK_FAIL, "Cannot handle such GridLocation [%s]"),
    ("U0110", CGM.CHECK_FAIL, "Cannot handle such ElementType [%s]"),
    ("S0300", CGM.CHECK_FAIL, "FamilyName is empty"),
    ("S0301", CGM.CHECK_FAIL, "Reference to unknown family [%s]"),
    ("S0302", CGM.CHECK_FAIL, "Reference to unknown additional family [%s]"),
    ("S0602", CGM.CHECK_FAIL, "Zone has no GridCoordinates"),
    ("S0603", CGM.CHECK_FAIL, "No GridCoordinates_t of name GridCoordinates in zone"),
    ("S0604", CGM.CHECK_WARN, "ZoneBC has no BC"),
)

USER_MESSAGES = {}
for k, l, m in messagetable:
    USER_MESSAGES[k] = (l, m)


# -----------------------------------------------------------------------------
class CGNS_VAL_USER_Checks(CGNS.VAL.parse.generic.GenericParser):
    def __init__(self, log):
        CGNS.VAL.parse.generic.GenericParser.__init__(self, log)
        self.log.addMessages(USER_MESSAGES)

    # --------------------------------------------------------------------
    def Zone_t(self, pth, node, parent, tree, log):
        rs = CGM.CHECK_OK
        zt = CGU.hasChildName(node, CGK.ZoneType_s)
        zv = []
        if zt is not None:
            if CGU.stringValueMatches(zt, CGK.Structured_s):
                cd = self.context[CGK.CellDimension_s][pth]
                self.context[CGK.IndexDimension_s][pth] = cd
            elif CGU.stringValueMatches(zt, CGK.Unstructured_s):
                self.context[CGK.IndexDimension_s][pth] = 1
            shp = (self.context[CGK.IndexDimension_s][pth], 3)
            if CGU.getShape(node) != shp:
                rs = log.push(pth, "S0009", CGU.getShape(node))
            elif CGU.stringValueMatches(zt, CGK.Structured_s):
                zd = node[1]
                for nd in range(self.context[CGK.IndexDimension_s][pth]):
                    zv.append(zd[nd][0])
                    if (zd[nd][1] != zd[nd][0] - 1) or (zd[nd][2] != 0):
                        rs = log.push(pth, "S0010")
            else:
                zv.append(node[1][0][0])
            self.context[CGK.VertexSize_s][pth] = tuple(zv)
        if CGU.hasChildNodeOfType(node, CGK.FamilyName_ts):
            basepath = [CGK.CGNSTree_ts, parent[0], node[0]]
            searchpath = basepath + [CGK.FamilyName_ts]
            famlist1 = CGU.getAllNodesByTypeOrNameList(tree, searchpath)
            searchpath = basepath + [CGK.AdditionalFamilyName_ts]
            famlist2 = CGU.getAllNodesByTypeOrNameList(tree, searchpath)
            for famlist, diagmessage in ((famlist1, "S0301"), (famlist2, "S0302")):
                for fampath in famlist:
                    famdefinition = CGU.getNodeByPath(tree, fampath)
                    if famdefinition[1] is None:
                        rs = log.push(pth, "S0300")
                    else:
                        famtarget = famdefinition[1].tostring().rstrip()
                        famtargetpath = "/%s/%s" % (parent[0], famtarget)
                        if famtargetpath not in self.context:
                            famtargetnode = CGU.getNodeByPath(tree, famtargetpath)
                            if famtargetnode is None:
                                rs = log.push(pth, diagmessage, famtarget)
                            else:
                                self.context[famtargetpath][pth] = True
        if not CGU.hasChildType(node, CGK.GridCoordinates_ts):
            rs = log.push(pth, "S0602")
        elif not CGU.hasChildName(node, CGK.GridCoordinates_s):
            rs = log.push(pth, "S0603")
        if not CGU.hasChildType(node, CGK.ZoneBC_ts):
            rs = log.push(pth, "S0604")
        return rs

    # --------------------------------------------------------------------
    def ZoneType_t(self, pth, node, parent, tree, log):
        rs = CGM.CHECK_OK
        if not CGU.stringValueInList(node, CGK.ZoneType_l):
            rs = log.push(pth, "S0101")
        return rs

    # --------------------------------------------------------------------
    def CGNSBase_t(self, pth, node, parent, tree, log):
        rs = CGM.CHECK_OK
        (cd, pd) = (0, 0)
        if not CGU.hasChildNodeOfType(node, CGK.Zone_ts):
            rs = log.push(pth, "U0101")
        else:
            target = [CGK.CGNSTree_ts, node[0], CGK.Zone_ts, CGK.ZoneType_s]
            plist = CGU.getAllNodesByTypeOrNameList(tree, target)
            found = False
            for p in plist:
                if CGU.stringValueMatches(CGU.getNodeByPath(tree, p), CGK.Structured_s):
                    found = True
            if not found:
                rs = log.push(pth, "U0102")
        if not CGU.hasChildNodeOfType(node, CGK.ReferenceState_ts):
            rs = log.push(pth, "U0103")
        if CGU.getShape(node) != (2,):
            rs = log.push(pth, "S0009", CGU.getShape(node))
        else:
            cd = node[1][0]
            pd = node[1][1]
            allowedvalues = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
            if (cd, pd) not in allowedvalues:
                rs = log.push(pth, "S0010", (cd, pd))
        self.context[CGK.CellDimension_s] = cd
        self.context[CGK.PhysicalDimension_s] = pd
        return rs

    # --------------------------------------------------------------------
    def GridLocation_t(self, pth, node, parent, tree, log):
        rs = CGM.CHECK_OK
        val = node[1].tostring()
        if val not in [CGK.Vertex_s, CGK.CellCenter_s, CGK.FaceCenter_s]:
            rs = log.push(pth, "U0109", val)
        return rs


# -----
