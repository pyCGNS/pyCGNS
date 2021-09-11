#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnstypes as CGT
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnserrors as CGE
import CGNS.VAL.parse.messages as CGM

import inspect

genericmessages = {
    "G0001": (CGM.CHECK_FAIL, "CGNSLibraryVersion [%s] is too old wrt check level"),
    "G0002": (CGM.CHECK_FAIL, "CGNSLibraryVersion is incorrect"),
    "G0003": (CGM.CHECK_FAIL, "Name [%s] is not valid"),
    "G0004": (CGM.CHECK_FAIL, "Name [%s] is a duplicated child name"),
    "G0005": (CGM.CHECK_FAIL, "PANIC: Cannot find node with path [%s]"),
    "G0006": (CGM.CHECK_FAIL, "PANIC: Node data is not numpy.ndarray or None"),
    "G0007": (CGM.CHECK_FAIL, "PANIC: Node children is not a list"),
    "G0008": (CGM.CHECK_FAIL, "PANIC: Node name is not a string"),
    "G0009": (CGM.CHECK_FAIL, "PANIC: Node is not a list of 4 objects"),
    "G0010": (CGM.CHECK_FAIL, "PANIC: Node is empty list or None"),
    "G0011": (CGM.CHECK_FAIL, "PANIC: Node name is empty string"),
    "G0012": (CGM.CHECK_FAIL, "PANIC: Node name has forbidden chars"),
    "G0013": (CGM.CHECK_FAIL, "PANIC: Node name is . or .."),
    "G0014": (CGM.CHECK_FAIL, "PANIC: Node name is too long"),
    "G0015": (CGM.CHECK_FAIL, "Bad node value data type"),
    "S0001": (CGM.CHECK_FAIL, "Unknown SIDS type [%s]"),
    "S0002": (CGM.CHECK_FAIL, "SIDS type [%s] not allowed as child of [%s]"),
    "S0003": (CGM.CHECK_FAIL, "SIDS type [%s] not allowed for this node"),
    "S0004": (CGM.CHECK_FAIL, "DataType [%s] not allowed for this node"),
    "S0005": (CGM.CHECK_FAIL, "Node [%s] of type [%s] not allowed as child"),
    "S0006": (CGM.CHECK_FAIL, "Node [%s] of type [%s] allowed only once as child"),
    "S0007": (CGM.CHECK_FAIL, "Node [%s] of type [%s] is mandatory"),
    "S0008": (CGM.CHECK_FAIL, "Child name [%s] reserved for a type in [%s]"),
    "S0009": (CGM.CHECK_FAIL, "Bad node shape [%s]"),
    "S0010": (CGM.CHECK_FAIL, "Bad node value"),
}


class PathContext(dict):
    def __getitem__(self, path):
        if path not in self:
            return self.scope(path)
        return dict.__getitem__(self, path)

    def scope(self, path):
        if path in self:
            return dict.__getitem__(self, path)
        level = 1
        while True:
            apath = CGU.getPathAncestor(path, level)
            if apath in self:
                return dict.__getitem__(self, apath)
            if apath == "/":
                return None
            level += 1


class GenericContext(dict):
    def __getitem__(self, key):
        if key not in self:
            dict.__setitem__(self, key, PathContext())
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        d = self[key]
        d["/"] = value


class GenericParser(object):
    # --------------------------------------------------------------------
    def __init__(self, log=None):
        self._stop = False
        self.keywordlist = CGK.cgnsnames
        if log is None:
            self.log = CGM.DiagnosticLog()
        self.log.addMessages(genericmessages)
        self.context = GenericContext()
        self._trace = False

    # --------------------------------------------------------------------
    def dbg(self, msg, *args):
        if self._trace:
            print(
                "> %-24s" % msg,
            )
            if args:
                for a in args:
                    print(
                        a,
                    )
            print("")

    # --------------------------------------------------------------------
    def listDiagnostics(self):
        return self.log.listMessages()

    # --------------------------------------------------------------------
    def checkLeafStructure(self, T, path, node, parent):
        stt = CGM.CHECK_GOOD
        try:
            CGU.checkNode(node, dienow=True)
            CGU.checkNodeName(node, dienow=True)
            CGU.checkDuplicatedName(parent, node[0], dienow=True)
            CGU.checkNodeType(node, dienow=True)
            if node[1] is not None:
                CGU.checkArray(node[1], dienow=True)
        except CGE.cgnsException as v:
            if v.code == 1:
                stt = self.log.push(path, "G0010")
            if v.code == 2:
                stt = self.log.push(path, "G0009")
            if v.code == 3:
                stt = self.log.push(path, "G0008")
            if v.code == 4:
                stt = self.log.push(path, "G0007")
            if v.code == 5:
                stt = self.log.push(path, "G0006")
            if v.code == 22:
                stt = self.log.push(path, "G0008")
            if v.code == 23:
                stt = self.log.push(path, "G0011")
            if v.code == 24:
                stt = self.log.push(path, "G0012")
            if v.code == 25:
                stt = self.log.push(path, "G0014")
            if v.code == 29:
                stt = self.log.push(path, "G0013")
            if v.code == 31:
                stt = self.log.push(path, "G0011")
            if v.code == 32:
                stt = self.log.push(path, "G0011")
            if v.code == 111:
                stt = self.log.push(path, "G0015")
            if self._stop:
                raise CGE.cgnsException(-1)
        return stt

    # --------------------------------------------------------------------
    def checkLeaf(self, T, path, node):
        parent = CGU.getParentFromNode(T, node)
        status1 = self.checkSingleNode(T, path, node, parent)
        status2 = status1
        ntype = CGU.getTypeAsGrammarToken(node[3])
        if (len(node) == 4) and (ntype in self.methods):
            status2 = getattr(self, ntype)(*[path, node, parent, T, self.log])
        else:
            # if (ntype in CGK.cgnstypes): print '\nSKIP ',ntype
            pass
        status1 = CGM.getWorst(status1, status2)
        if self._stop and status1 == CGM.CHECK_FAIL:
            raise CGE.cgnsException(-1)
        return status1

    # --------------------------------------------------------------------
    def checkSingleNode(self, T, path, node, parent):
        stt = CGM.CHECK_GOOD
        if not CGU.checkNodeName(node):
            stt = self.log.push(path, "G0003", node[0])
        lchildren = CGU.childNames(parent)
        if lchildren:
            if node[0] in lchildren:
                lchildren.remove(node[0])
            if node[0] in lchildren:
                stt = self.log.push(path, "G0004", node[0])
        tlist = CGU.getNodeAllowedChildrenTypes(parent, node)
        if (CGU.getTypeAsGrammarToken(node[3]) not in tlist) and (
            node[3] != CGK.CGNSTree_ts
        ):
            if parent is not None:
                stt = self.log.push(path, "S0002", node[3], parent[3])
            else:
                stt = self.log.push(path, "S0003", node[3])
        dlist = CGU.getNodeAllowedDataTypes(node)
        dt = CGU.getValueDataType(node)
        if dt not in dlist:
            stt = self.log.push(path, "S0004", dt)
        if node[3] not in CGT.types.keys():
            stt = self.log.push(path, "S0001", node[3])
        else:
            stt = self.checkCardinalityOfChildren(T, path, node, parent)
            stt = self.checkReservedChildrenNames(T, path, node, parent)
        return stt

    # --------------------------------------------------------------------
    def checkTreeStructure(self, T, path=""):
        status = CGM.CHECK_GOOD
        status = self.checkLeafStructure(T, path, T, None)
        if status == CGM.CHECK_GOOD:
            path = path + "/" + T[0]
            for c in T[2]:
                status = self.checkTreeStructure(c, path)
        return status

    # --------------------------------------------------------------------
    def checkTree(self, T, trace=False, stop=False):
        self._stop = stop
        self._trace = trace
        status1 = CGM.CHECK_GOOD
        if self._trace:
            print("### Parsing node paths...")
        status1 = self.checkTreeStructure(T)
        if status1 != CGM.CHECK_GOOD:
            return status1
        paths = CGU.getPathFullTree(T, width=True)
        sz = len(paths) + 1
        ct = 1
        if not hasattr(self, "methods"):
            self.methods = []
            for m in inspect.getmembers(self):
                if (m[0][-2:] == "_t") or (m[0][-2:] == "_n") or (m[0][-3:] == "_ts"):
                    self.methods += [m[0]]
        for path in ["/"] + paths:
            if self._trace:
                print(
                    "### Check node [%.6d/%.6d]\r" % (ct, sz),
                )
            node = CGU.getNodeByPath(T, path)
            status2 = CGM.CHECK_GOOD
            if node is None:
                status2 = self.log.push(path, "G0005", path)
                if self._stop:
                    raise CGE.cgnsException(-1)
            if status2 == CGM.CHECK_GOOD:
                status2 = self.checkLeaf(T, path, node)
            status1 = status2
            ct += 1
        if self._trace:
            print("")
        return status1

    # --------------------------------------------------
    def checkCardinalityOfChildren(self, T, path, node, parent):
        stt = CGM.CHECK_GOOD
        for child in node[2]:
            card = CGT.types[node[3]].cardinality(child[3])
            if card == CGT.C_00:
                if path == "/":
                    cpath = "/%s" % (child[0])
                else:
                    cpath = "%s/%s" % (path, child[0])
                stt = self.log.push(path, "S0005", cpath, child[3])
            if card in [CGT.C_11, CGT.C_01]:
                if [c[3] for c in node[2]].count(child[3]) > 1:
                    stt = self.log.push(path, "S0006", child[0], child[3])
        for tchild in CGT.types[node[3]].children:
            card = CGT.types[node[3]].cardinality(tchild[0])
            if card in [CGT.C_11, CGT.C_1N]:
                if [c[3] for c in node[2]].count(tchild[0]) < 1:
                    stt = self.log.push(path, "S0007", tchild[1][0], tchild[0])
        return stt

    # --------------------------------------------------
    def checkReservedChildrenNames(self, T, path, node, parent):
        stt = CGM.CHECK_GOOD
        for child in node[2]:
            rt = CGT.types[node[3]].hasReservedNameType(child[0])
            if (rt != []) and (child[3] not in rt):
                srt = ""
                for s in rt:
                    srt = srt + "," + s
                stt = self.log.push(path, "S0008", child[0], srt[1:])
        return stt

    # --------------------------------------------------------------------
    def CGNSLibraryVersion_t(self, pth, node, parent, tree, log):
        stt = CGM.CHECK_OK
        try:
            version = int(node[1][0] * 1000)
            if version < 2400:
                stt = self.log.push(pth, "G0001", version)
            if version > 3400:
                stt = self.log.push(pth, "G0002")
        except Exception:
            stt = self.log.push(pth, "G0002")
        return stt


# --- last line
