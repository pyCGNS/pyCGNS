#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import string

CHECK_NONE=0
CHECK_GOOD=1
CHECK_WARN=2
CHECK_FAIL=3
CHECK_USER=4

INVALID_NAME='Name [%s] is not valid'
DUPLICATED_NAME='Name [%s] is a duplicated child name'
INVALID_SIDSTYPE_P='SIDS Type [%s] not allowed as child of [%s]'
INVALID_SIDSTYPE='SIDS Type [%s] not allowed for this node'
INVALID_DATATYPE='Datatype [%s] not allowed for this node'

def getWorst(st1,st2):
    if (CHECK_FAIL in [st1,st2]): return CHECK_FAIL
    if (CHECK_USER in [st1,st2]): return CHECK_USER
    if (CHECK_WARN in [st1,st2]): return CHECK_WARN
    if (CHECK_GOOD in [st1,st2]): return CHECK_GOOD
    return CHECK_NONE
    
class DiagnosticLog(dict):
    __diagstr={CHECK_FAIL:'#',CHECK_USER:'%',
               CHECK_WARN:'!',CHECK_GOOD:'=',CHECK_NONE:'?'}
    def push(self,path,level,message):
        if (path not in self): self[path]=[]
        self[path].append((level,message))
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
    def pline(self,path,entry):
        shft=self.shift(path)
        s =shft+self.asStr(entry[0])+path+'\n'
        s+=shft+self.asStr(entry[0])+self.asStr(entry[0])+'\n'
        s+=shft+self.asStr(entry[0])+entry[1]
        return s
    def diagForPath(self,path):
        if (path in self): return self[path]
        return None
    def diagnostics(self,path):
        if (self.diagForPath(path) is not None):
            for diag in self[path]:
                yield diag
# --- last line
