#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#

globalTraceFlag=0
globalLeaveFlag=0

def ttt(msg):
  global globalTraceFlag
  if (globalTraceFlag):  print "### C5 [Trace]: %s"%(msg)

def trace(flag=1):
  global globalTraceFlag
  globalTraceFlag=flag

def leave(flag=1):
  global globalLeaveFlag
  globalLeaveFlag=flag

def error(code,msg,kill=1):
  print "### C5: Error - [%.3d] %s"%(code,msg)
  if (kill):
    c5exit()

def pstack(e):
  n=0
  for l in e:
    n+=1
    print "### C5: %.2d %s"%(n,l[:-1])
    
def c5exit():
  global globalLeaveFlag
  if (globalLeaveFlag):
    raise c5Exception()
  else:
    import sys
    sys.exit(-1)

class c5Exception(Exception):
  def __init__(self):
    ttt("Exception in C5")

def message(msg):
  print "### C5: %s"%msg
