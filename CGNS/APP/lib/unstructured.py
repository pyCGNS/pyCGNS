import CGNS.MAP              as CGM
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils    as CGU
import CGNS.PAT.cgnslib      as CGL

import CGNS.APP.probe.arrayutils as ARU


def parseTree(filename):
    flags = CGM.S2P_DEFAULT
    (tree, l) = CGM.load(filename, flags, 0, None, [], None)
    typepath = [CGK.CGNSTree_ts, CGK.CGNSBase_ts, CGK.Zone_ts, CGK.Elements_ts]
    elist = CGU.getAllNodesByTypeList(tree, typepath)
    sn = 0
    sl = []
    sp = ARU.SectionParse()
    mr = 1
    for e in elist:
        print('Parse ', e)
        sn += 1
        ne = CGU.getNodeByPath(tree, e)[1]
        et = ne[0]
        eb = ne[1]
        ea = CGU.getNodeByPath(tree, e + '/' + CGK.ElementConnectivity_s)[1]
        if (et in sp.QUAD_SURFACE):
            sl.append(sp.extQuadFacesPoints(ea, et, sn, mr, eb))
        if (et in sp.TRI_SURFACE):
            sl.append(sp.extTriFacesPoints(ea, et, sn, mr, eb))
        mr = sl[-1][-1]

# for s in sl:
#       print CGK.ElementType_l[s[0]]
#       if (s[0]==CGK.QUAD_4):
#         for p in range(len(s[1])/4):
#             print s[1][4*p+0],s[1][4*p+1],s[1][4*p+2],s[1][4*p+3]

# --- last line
