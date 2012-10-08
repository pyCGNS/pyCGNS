#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import string

CHECK_NONE=0
CHECK_OK=CHECK_GOOD=1
CHECK_INFO=CHECK_WARN=2
CHECK_BAD=CHECK_ERROR=CHECK_FAIL=3
CHECK_USER=4


def getWorst(st1,st2):
    if (CHECK_FAIL in [st1,st2]): return CHECK_FAIL
    if (CHECK_USER in [st1,st2]): return CHECK_USER
    if (CHECK_WARN in [st1,st2]): return CHECK_WARN
    if (CHECK_GOOD in [st1,st2]): return CHECK_GOOD
    return CHECK_NONE
    
class DiagnosticLog(dict):
    __diagstr={CHECK_FAIL:'E',CHECK_USER:'U',
               CHECK_WARN:'W',CHECK_GOOD:' ',CHECK_NONE:'?'}
    __messages={}
    def __init__(self):
        dict.__init__(self)
        DiagnosticLog.__messages
    def noContextMessage(self,m):
        if ('%' in DiagnosticLog.__messages[m]): return None
        return DiagnosticLog.__messages[m]
    def addMessages(self,d):
        for e in d:
            DiagnosticLog.__messages[e]=d[e]
    def push(self,path,level,messagekey,*tp):
        if (path not in self): self[path]=[]
        if (messagekey not in DiagnosticLog.__messages): return
        msg=DiagnosticLog.__messages[messagekey]
        try:
            if (tp): msg=msg%tp
        except TypeError:
            pass
        self[path].append((level,msg,messagekey))
        return level
    def __len__(self):
        return len(self.keys())
    def shift(self,path,shiftstring=' '):
        n=string.split(path,'/')
        return len(n)*shiftstring
    def getWorst(self,st1,st2):
        return getWorst(st1,st2)
    def asStr(self,st):
        if (st in self.__diagstr): return self.__diagstr[st]
        return self.__diagstr[CHECK_NONE]
    def status(self,entry):
        return entry[0]
    def hasOnlyKey(self,path,keylist):
        k1=set(keylist)
        k2=set([e[2] for e in self[path]])
        return k2.issubset(k1)
    def key(self,entry):
        return entry[2]
    def message(self,entry,path=None):
        shft=''
        if (path is not None): shft=self.shift(path)
        s='%s[%s:%s] %s'%(shft,entry[2],self.asStr(entry[0]),entry[1])
        return s
    def getWorstDiag(self,path):
        s=set([e[2] for e in self[path]])
        r=reduce(getWorst,s)
        print r
        return r
    def diagForPath(self,path):
        if (path in self): return self[path]
        return None
    def allMessageKeys(self):
        r=set()
        for path in self:
            for entry in self[path]:
                r.add(entry[2])
        mlist=list(r)
        mlist.sort()
        return mlist
    def allPathKeys(self):
        return self.keys()
    def diagnosticsByPath(self,path):
        if (self.diagForPath(path) is not None):
            for diag in self[path]:
                yield (diag,path)
    def diagnosticsByMessage(self,msg):
        plist=self.keys()
        plist.sort()
        for path in plist:
            for diag in self[path]:
                if (diag[2]==msg): yield (diag,path)
# --- last line
