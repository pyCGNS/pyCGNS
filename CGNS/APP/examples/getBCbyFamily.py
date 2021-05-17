import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK


def getBCbyFamily(T, family):
    """Finds all BC that belongs to a family.
    Works with more than one base.
    Returns a list of path, you have to use getNodeByPath to retrieve
    the actual node."""
    result = []
    fpath = [
        CGK.CGNSTree_ts,
        CGK.CGNSBase_t,
        CGK.Zone_t,
        CGK.ZoneBC_t,
        CGK.BC_t,
        CGK.FamilyName_t,
    ]
    flist = CGU.getAllNodesByTypeOrNameList(T, fpath)
    for famnamepath in flist:
        fnode = CGU.getNodeByPath(T, famnamepath)
        if CGU.stringValueMatches(fnode, family):
            bcpath = CGU.getAncestor(famnamepath)
            result.append(bcpath)
    return result
