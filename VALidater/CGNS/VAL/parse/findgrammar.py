#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

import sys
import os
import fnmatch
import imp

def findAllUserGrammars(verbose=False):
  kdict={}
  for pth in sys.path:
    if (verbose): print '### CGNS.VAL: scanning',pth
    try:
      for pthroot, dirs, files in os.walk(pth):
          for fn in files:
              if fnmatch.fnmatch(fn,'CGNS_VAL_USER_*.py'):
                  kdict[fn[14:-3]]=pthroot
    except OSError: pass
  return kdict

def importUserGrammars(key,verbose=False):
  mod=None
  modname='CGNS_VAL_USER_%s'%key
  try:
      tp=imp.find_module(modname)
  except ImportError:
      dk=findAllUserGrammars()
      if (key in dk):
          sys.path.append(dk[key])
          print '### CGNS.VAL [warning]: not in search path [%s]\n'%dk[key]
          try:
              tp=imp.find_module(modname)
          except ImportError:
              return None
      else:
          return None
  try:
    fp=tp[0]
    mod=imp.load_module(modname, *tp)
  finally:
    if fp:
       fp.close()
  return mod

# ---
