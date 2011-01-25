#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release: v4.0.1 $
#  -------------------------------------------------------------------------

CGNSPythonGlobalTreeDict={'/foo/bar/zap':None}
CGNSPythonGlobalContext={}

NOERROR=0

# ====================================================================
def getContext(id):
  for i in CGNSPythonGlobalContext:
    if (i.id==id):
      return CGNSPythonGlobalContext[i]
  return None

def getKey(id):
  for i in CGNSPythonGlobalContext:
    if (i.id==id):
      return i
  return None

# ====================================================================
cdef extern from "cgnslib.h":

 # ------------------------------------------------------------
 int cg_open(char *filename, int mode, int *fn):
   if (filename in CGNSPythonGlobalTreeDict):
     CGNSPythonGlobalContext[filename]=CGNSPythonContext(filename)
   fn[0]=CGNSPythonGlobalContext[filename].id
   return NOERROR

 # ------------------------------------------------------------
 int cg_close(int fn):
   key=getKey(id)
   if (key):
     del CGNSPythonGlobalContext[key]
   return NOERROR 

 int cg_version(int fn, float *version)
 int cg_nbases(int fn, int *nbases)

# ====================================================================
class CGNSPythonContext:
  count=0
  def __init__(self):
    self.currentNode=None
    self.id=CGNSPythonContext.count
    self.error=-1
    CGNSPythonContext.count+=1


# ====================================================================
