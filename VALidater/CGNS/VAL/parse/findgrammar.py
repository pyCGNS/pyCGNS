#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import sys
import os
import fnmatch
import imp

PROFILENAME='grammars'

#  -------------------------------------------------------------------------
def readProfile():
  try:
    hdir=os.environ['HOME']
  except:
    return {}
  pdir='%s/.CGNS.NAV'%hdir
  if (not os.exists(pdir)):
    return {}
  sys.path.append(pdir)
  fp, pth, des = imp.find_module('grammars')
  try:
    mod=imp.load_module('grammars', fp, pth, des)
  finally:
    if fp:
      fp.close()
  return mod.Grammars
  
#  -------------------------------------------------------------------------
def findAllUserGrammars(verbose=False):
  kdict={}
  for pth in [p for p in sys.path if p!='']:
    if (verbose): print '### scanning',pth
    try:
      for pthroot, dirs, files in os.walk(pth):
          for fn in files:
              if (fnmatch.fnmatch(fn,'CGNS_VAL_USER_*.py')
                  or fnmatch.fnmatch(fn,'CGNS_VAL_USER_*.so')):
                gkey=fn[14:-3]
                if (gkey in kdict):
                  if (verbose):
                    print '### * found grammar:',
                    print gkey,'already found, ignore this one'
                else:
                  if (verbose):
                    print '### * found grammar:',gkey
                  kdict[fn[14:-3]]=pthroot
                if (verbose):
                  print '### * found in :',pthroot
                  if (pthroot not in sys.path): 
                    print '### * previous path is NOT in PYTHONPATH'
                  else:
                    print '### * previous path already is in PYTHONPATH'
                    
    except OSError: pass
  return kdict

#  -------------------------------------------------------------------------
def findOneUserGrammar(tag,verbose=False):
  kdict={}
  found=False
  for pth in sys.path:
    if (verbose): print '### scanning',pth
    try:
      for pthroot, dirs, files in os.walk(pth):
          for fn in files:
              if fnmatch.fnmatch(fn,'CGNS_VAL_USER_%s.py'%tag):
                  kdict[fn[14:-3]]=pthroot
                  found=True
                  break
          if found: break
    except OSError: pass
    if found: break
  return kdict

#  -------------------------------------------------------------------------
def importUserGrammars(key,recurse=False,verbose=False):
  mod=None
  modname='CGNS_VAL_USER_%s'%key
  prog=sys.argv[0]
  dpref='./'
  if (prog[0]=='/'):
    dpref=os.path.dirname(prog)
  else:
    for pp in os.environ['PATH'].split(':'):
      if (os.path.exists("%s/%s"%(pp,prog))):
        dpref=os.path.dirname("%s/%s"%(pp,prog))
        break
  ppref=dpref+'/../lib/python%s/site-packages'%sys.version[:3]
  ppref=os.path.normpath(os.path.abspath(ppref))
  if (ppref+'/CGNS/PRO' not in sys.path):
    sys.path.append(ppref+'/CGNS/VAL/grammars')
  if (ppref+'/CGNS/PRO' not in sys.path):
    sys.path.append(ppref+'/CGNS/PRO')
  ipath='%s/lib/python%s.%s/site-packages/CGNS/PRO'%\
         (sys.prefix,sys.version_info[0],sys.version_info[1])
  sys.path.append(ipath)
  ipath='%s/lib/python%s.%s/site-packages/CGNS/VAL/grammars'%\
         (sys.prefix,sys.version_info[0],sys.version_info[1])
  sys.path.append(ipath)

  print sys.path
  if (verbose): print '### Looking for grammar [%s]'%key
  try:
    tp=imp.find_module(modname)
  except ImportError:
    if (verbose): print '### Error: grammar [%s] not found'%key
    if (recurse):
      dk=findOneUserGrammar(key)
      if (key in dk):
          sys.path.append(dk[key])
          if (verbose): print '### Warning: not in search path [%s]'%dk[key]
          try:
              tp=imp.find_module(modname)
          except ImportError:
              return None
      else:
          return None
    else:
      return None
  try:
    fp=tp[0]
    if (tp[2][2]!=imp.C_EXTENSION):
      mod=imp.load_module(modname, *tp)
    else:
      #print '### CGNS.VAL [info]: Module info',tp
      mod=imp.load_dynamic(modname,tp[1],tp[0])
  except:
    pass
  finally:
    if fp:
       fp.close()
  return mod

# ---
