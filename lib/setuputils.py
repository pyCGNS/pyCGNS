#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
MAJORVERSION=4
MINORVERSION=5
REVISION=0
# --------------------------------------------------------------------

import os
import sys
import shutil
import re
import time
import subprocess
import distutils.util

from distutils.dir_util import remove_tree
from distutils.core import setup
from distutils.util import get_platform
from distutils.command.clean import clean as _clean

rootfiles=['__init__.py','errors.py','version.py','config.py','test.py']
compfiles=['midlevel.py','wrap.py']

pfx='# '

# if you change this name, also change lines tagged with 'USER CONFIG'
userconfigfile='setup_userConfig.py'

class ConfigException(Exception):
  pass

# --------------------------------------------------------------------
def prodtag():
  from time import gmtime, strftime
  proddate=strftime("%Y-%m-%d %H:%M:%S", gmtime())
  try:
    prodhost=os.uname()
  except AttributeError:
    prodhost='win32'
  return (proddate,prodhost)

# http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

# --------------------------------------------------------------------
def unique_but_keep_order(lst):
  if (len(lst)<2): return lst
  r=[lst[0]]
  for p in lst[1:]:
    if (p not in r): r.append(p)
  return r
    
# --------------------------------------------------------------------
def search(incs,libs,tag='pyCGNS',
           deps=['Cython','HDF5','MLL','numpy','vtk','CHLone',
                 'PySide','SQLAlchemy']):
  state=1
  for com in sys.argv:
    if com in ['help','clean']: state=0
  pt=distutils.util.get_platform()
  vv="%d.%d"%(sys.version_info[0],sys.version_info[1])
  tg="%s/./build/lib.%s-%s/CGNS"%(os.getcwd(),pt,vv)
  bptarget=tg
  if (not os.path.exists(bptarget)): os.makedirs(bptarget)
  oldsyspath=sys.path
  sys.path =[os.path.abspath(os.path.normpath('./lib'))]
  cfgdict={}
  import pyCGNSconfig_default as C_D
  sys.path=oldsyspath
  for ck in dir(C_D):
    if (ck[0]!='_'): cfgdict[ck]=C_D.__dict__[ck]
  pg=prodtag()
  cfgdict['PFX']=pfx
  cfgdict['DATE']=pg[0]
  cfgdict['PLATFORM']="%s %s %s"%(pg[1][0],pg[1][1],pg[1][-1])
  updateConfig('..',bptarget,cfgdict)
  sys.path=[bptarget]+sys.path

  # here we go, check each dep and add incs/libs/others to config
  try:
    import pyCGNSconfig as C

    # -----------------------------------------------------------------------
    if ('Cython' in deps):
      try:
        if (which('cython') is not None):
          C.COM_CYTHON='cython'
        else:
          raise Exception
        if (which('pyside-uic') is not None): C.COM_UIC='pyside-uic'
        if (which('pyside-rcc') is not None): C.COM_RCC='pyside-rcc'
        if (which('pyrcc4') is not None):     C.COM_RCC='pyrcc4'
        import Cython
        C.HAS_CYTHON=True
        print pfx+'using Cython v%s'%Cython.__version__
        C.HAS_CYTHON_2PLUS=False
        try:
          if (float(Cython.__version__[:3])>0.1):
            C.HAS_CYTHON_2PLUS=True
          else:
            print pfx+'***** warning Cython version cannot build CGNS/WRA'
        except:
          print pfx+'***** warning Cython version cannot build CGNS/WRA'
      except:
        C.HAS_CYTHON=False
        print pfx+'FATAL: Cython not found'

    # -----------------------------------------------------------------------
    if ('PySide' in deps):
      try:
        import PySide
        import PySide.QtCore
        import PySide.QtGui
        
        C.HAS_PYSIDE=True
        print pfx+'using PySide v%s'%PySide.__version__
      except:
        C.HAS_PYSIDE=False
        print pfx+'ERROR: PySide not found'

    # -----------------------------------------------------------------------
    if ('vtk' in deps):
      try:
        import vtk
        v=vtk.vtkVersion()
        C.HAS_VTK=True
        print pfx+'using vtk (python module) v%s'%v.GetVTKVersion()
      except:
        C.HAS_VTK=False
        print pfx+'ERROR: no vtk python module'

    # -----------------------------------------------------------------------
    if ('numpy' in deps):
      incs=incs+C.NUMPY_PATH_INCLUDES
      libs=libs+C.NUMPY_PATH_LIBRARIES
      tp=find_numpy(incs,libs,C.NUMPY_LINK_LIBRARIES)
      if (tp is None):
        print pfx+'FATAL: setup cannot find Numpy'
        sys.exit(1)
      (C.NUMPY_VERSION,
       C.NUMPY_PATH_INCLUDES,
       C.NUMPY_PATH_LIBRARIES,
       C.NUMPY_LINK_LIBRARIES,
       C.NUMPY_EXTRA_ARGS)=tp
      print pfx+'using Numpy API version %s'%(C.NUMPY_VERSION,)
      print pfx+'using Numpy headers from %s'%(C.NUMPY_PATH_INCLUDES[0])
      C.HAS_NUMPY=True
      incs=incs+C.NUMPY_PATH_INCLUDES
      libs=libs+C.NUMPY_PATH_LIBRARIES

    # -----------------------------------------------------------------------
    if ('HDF5' in deps):
      incs=incs+C.HDF5_PATH_INCLUDES+C.INCLUDE_DIRS
      libs=libs+C.HDF5_PATH_LIBRARIES+C.LIBRARY_DIRS
      tp=find_HDF5(incs,libs,C.HDF5_LINK_LIBRARIES)
      if (tp is None):
        print pfx+'ERROR: setup cannot find HDF5!'
        sys.exit(1)
      (C.HDF5_VERSION,
       C.HDF5_PATH_INCLUDES,
       C.HDF5_PATH_LIBRARIES,
       C.HDF5_LINK_LIBRARIES,
       C.HDF5_EXTRA_ARGS)=tp
      print pfx+'using HDF5 %s'%(C.HDF5_VERSION,)
      print pfx+'using HDF5 headers from %s'%(C.HDF5_PATH_INCLUDES[0])
      print pfx+'using HDF5 libs from %s'%(C.HDF5_PATH_LIBRARIES[0])
      C.HAS_HDF5=True
      incs=incs+C.HDF5_PATH_INCLUDES+C.INCLUDE_DIRS
      libs=libs+C.HDF5_PATH_LIBRARIES+C.LIBRARY_DIRS

    # -----------------------------------------------------------------------
    if ('MLL' in deps):
      incs=incs+C.MLL_PATH_INCLUDES
      libs=libs+C.MLL_PATH_LIBRARIES
      tp=find_MLL(incs,libs,C.MLL_LINK_LIBRARIES,C.MLL_EXTRA_ARGS)
      if (tp is None):
        print pfx+'ERROR: setup cannot find cgns.org library (MLL)!'
        C.HAS_MLL=False
      else:
        (C.MLL_VERSION,
         C.MLL_PATH_INCLUDES,
         C.MLL_PATH_LIBRARIES,
         C.MLL_LINK_LIBRARIES,
         C.MLL_EXTRA_ARGS,ifound,lfound)=tp
        print pfx+'using CGNS/MLL %s'%(C.MLL_VERSION)
        print pfx+'using CGNS/MLL headers from %s'%ifound
        print pfx+'using CGNS/MLL libs from %s'%lfound
        C.HAS_MLL=True
      incs=incs+C.MLL_PATH_INCLUDES
      libs=libs+C.MLL_PATH_LIBRARIES
       
    # -----------------------------------------------------------------------
    if ('CHLone' in deps):
      try:
        import CHLone
        C.HAS_CHLONE=True
        print pfx+'using CHLone %s'%CHLone.version
        
      except:
        print pfx+'ERROR: setup cannot import CHLone!'
        C.HAS_CHLONE=False
       
    # -----------------------------------------------------------------------

  except ImportError:
    print pfx+'ERROR: setup cannot find pyCGNSconfig.py file!'
    sys.exit(1)
  C.MLL_PATH_INCLUDES=list(set(C.MLL_PATH_INCLUDES))
  C.MLL_PATH_LIBRARIES=list(set(C.MLL_PATH_LIBRARIES))
  C.HDF5_PATH_INCLUDES=list(set(C.HDF5_PATH_INCLUDES))
  C.HDF5_PATH_LIBRARIES=list(set(C.HDF5_PATH_LIBRARIES))
  C.CHLONE_PATH_INCLUDES=list(set(C.CHLONE_PATH_INCLUDES))
  C.CHLONE_PATH_LIBRARIES=list(set(C.CHLONE_PATH_LIBRARIES))
  C.NUMPY_PATH_INCLUDES=list(set(C.NUMPY_PATH_INCLUDES))
  C.NUMPY_PATH_LIBRARIES=list(set(C.NUMPY_PATH_LIBRARIES))

  incs=unique_but_keep_order(incs)
  libs=unique_but_keep_order(libs)

  C.INCLUDE_DIRS=incs
  C.LIBRARY_DIRS=libs

  C.PRODUCTION_DIR=bptarget
  
  updateConfig('..',bptarget,C.__dict__,cfgdict)

  return (C, state)

# --------------------------------------------------------------------
def installConfigFiles(bptarget):
  lptarget='.'
  for ff in rootfiles:
    shutil.copy("%s/lib/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff))
  for ff in compfiles:
    shutil.copy("%s/lib/compatibility/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff))

# --------------------------------------------------------------------
def updateVersionInFile(filename,bptarget):
  f=open('%s/revision.tmp'%bptarget)
  r=int(f.readlines()[0][:-1])
  REVISION=r
  f=open(filename,'r')
  l=f.readlines()
  f.close()
  vver='@@UPDATEVERSION@@'
  vrel='@@UPDATERELEASE@@'
  vrev='@@UPDATEREVISION@@'
  r=[]
  for ll in l:
    rl=ll
    if (ll[-len(vver)-1:-1]==vver): 
      rl='__version__=%s # %s\n'%(MAJORVERSION,vver)
    if (ll[-len(vrel)-1:-1]==vrel): 
      rl='__release__=%s # %s\n'%(MINORVERSION,vrel)
    if (ll[-len(vrev)-1:-1]==vrev):
      ACTUALREV=REVISION
      rl='__revision__=%s # %s\n'%(ACTUALREV,vrev)
    r+=[rl]
  f=open(filename,'w+')
  f.writelines(r)
  f.close()

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything
relist=['^.*~$','^core\.*$','^pyCGNS\.log\..*$',
        '^#.*#$','^.*\.aux$','^.*\.pyc$','^.*\.bak$','^.*\.l2h',
        '^Output.*$']
reclean=[]

for restring in relist:
  reclean.append(re.compile(restring))

def wselect(args,dirname,names):
  for n in names:
    for rev in reclean:
      if (rev.match(n)):
        # print "%s/%s"%(dirname,n)
        os.remove("%s/%s"%(dirname,n))
        break

class clean(_clean):
  def run(self):
    import glob
    rdirs=glob.glob("./build/*")
    for d in rdirs: remove_tree(d)
    if os.path.exists("./build"):     remove_tree("./build")
    if os.path.exists("./Doc/_HTML"): remove_tree("./Doc/_HTML")
    if os.path.exists("./Doc/_PS"):   remove_tree("./Doc/_PS")
    if os.path.exists("./Doc/_PDF"):  remove_tree("./Doc/_PDF")

# --------------------------------------------------------------------
def confValueAsStr(v):
  if (type(v)==type((1,))): return str(v)
  if (type(v)==type([])):   return str(v)
  if (v in [True,False]):   return str(v)
  else:                     return '"%s"'%str(v)

# --------------------------------------------------------------------
def updateConfig(pfile,gfile,config_default,config_previous=None):
  if (config_previous):
    from pyCGNSconfig_default import file_pattern as fpat
    cfg=config_default
    for ck in config_previous:
      if (not cfg.has_key(ck)): cfg[ck]=config_previous[ck]
    f=open("%s/pyCGNSconfig.py"%(gfile),'w+')
    f.writelines(fpat%cfg)
    f.close()
    return
  elif (not os.path.exists("%s/pyCGNSconfig.py"%(gfile))):
    print "### pyCGNS: create new pyCGNSconfig.py file"
    newconf=1
  else:
    f1=os.stat("%s/pyCGNSconfig.py"%(gfile))
    if (os.path.exists("%s/%s"%(pfile,userconfigfile))):
      f2=os.stat("%s/%s"%(pfile,userconfigfile))
    else:
      f2=os.stat("./%s"%userconfigfile)
    if (f1.st_mtime < f2.st_mtime):
      newconf=1
      print pfx+"using modified %s file"%userconfigfile
    else:
      newconf=0
      print pfx+"using existing %s file"%userconfigfile
  if newconf:
    sys.path=['..']+['.']+sys.path
    import setup_userConfig as UCFG # USER CONFIG
    for ck in dir(UCFG):
      if (ck[0]!='_'): config_default[ck]=UCFG.__dict__[ck]
    if (not os.path.exists('%s'%gfile)):
      os.makedirs('%s'%gfile)
    f=open("%s/pyCGNSconfig.py"%(gfile),'w+')
    f.writelines(config_default['file_pattern']%config_default)
    f.close()

# --------------------------------------------------------------------
def frompath_HDF5():
  try:
   h5p=subprocess.check_output(["which","h5dump"])
  except:
    try:
      h5p=subprocess.check_output(["whence","h5dump"])
    except:
      h5p=None
  if (h5p is not None):
    h5root='/'.join(h5p.split('/')[:-2])
  else:
    h5root='/usr/local'
  return h5root

# --------------------------------------------------------------------
def frompath_MLL():
  try:
   mllp=subprocess.check_output(["which","cgnscheck"],stderr=subprocess.STDOUT)
  except:
    try:
      mllp=subprocess.check_output(["whence","cgnscheck"],stderr=subprocess.STDOUT)
    except:
      mllp=None
  if (mllp is not None):
    mllroot='/'.join(mllp.split('/')[:-2])
  else:
    mllroot='/usr/local'
  return mllroot

# --------------------------------------------------------------------
def find_HDF5(pincs,plibs,libs):
  notfound=1
  extraargs=[]
  vers=''
  h5root=frompath_HDF5()
  pincs+=[h5root,'%s/include'%h5root]
  plibs+=[h5root,'%s/lib64'%h5root]
  plibs+=[h5root,'%s/lib'%h5root]
  pincs=unique_but_keep_order(pincs)
  plibs=unique_but_keep_order(plibs)
  for pth in plibs:
    if (    (os.path.exists(pth+'/libhdf5.a'))
         or (os.path.exists(pth+'/libhdf5.so'))
         or (os.path.exists(pth+'/libhdf5.sl'))):
      notfound=0
      plibs=[pth]
      break
  if notfound:
    print pfx+"ERROR: libhdf5 not found, please check paths:"
    for ppl in plibs:
      print pfx,ppl
  notfound=1
  for pth in pincs:
    if (os.path.exists(pth+'/hdf5.h')): notfound=0
  if notfound:
    print pfx,"ERROR: hdf5.h not found, please check paths"
    for ppi in pincs:
      print pfx,ppi
    return None

  ifh='HDF5 library version: unknown'
  notfound=1
  for pth in pincs:
    if (os.path.exists(pth+'/H5public.h')):
      fh=open(pth+'/H5public.h','r')
      fl=fh.readlines()
      fh.close()
      found=0
      for ifh in fl:
        if (ifh[:21] == "#define H5_VERS_INFO "):
          vers=ifh.split('"')[1].split()[-1]
          found=1
      if found:
        pincs=[pth]
        notfound=0
        break
  if notfound:
      print pfx,"ERROR: cannot find hdf5 version, please check paths"
      for ppi in pincs:
        print pfx,pincs
      return None

  return (vers,pincs,plibs,libs,extraargs)

# --------------------------------------------------------------------
def find_MLL(pincs,plibs,libs,extraargs):
  notfound=1
  vers=''
  cgnsversion='3200'
  mllroot=frompath_MLL()
  pincs+=[mllroot,'%s/include'%mllroot]
  plibs+=[mllroot,'%s/lib'%mllroot]
  libs=['cgns','hdf5']+libs
  pincs=unique_but_keep_order(pincs)
  plibs=unique_but_keep_order(plibs)
  extraargs=[]#'-DCG_BUILD_SCOPE']
  lfound=''
  ifound=''
  for pth in pincs:
    if (os.path.exists(pth+'/cgnslib.h')):
      notfound=0
      f=open(pth+'/cgnslib.h','r')
      l=f.readlines()
      f.close()
      for ll in l:
        if (ll[:20]=="#define CGNS_VERSION"):
          cgnsversion=ll.split()[2]
          if (cgnsversion<'3200'):
            print pfx,"ERROR: version should be v3.2 for MLL"
            return None
      ifound=pth
      break
  if notfound:
    print pfx+"ERROR: cgnslib.h not found, please check paths"
    for ppi in pincs:
      print pfx,ppi
    return None

  notfound=1
  for pth in plibs:
    if (    (os.path.exists(pth+'/libcgns.a'))
         or (os.path.exists(pth+'/libcgns.so'))
         or (os.path.exists(pth+'/libcgns.sl'))):
      cgnslib='cgns'
      notfound=0
      lfound=pth
      break
  if notfound:
    print pfx+"ERROR: libcgns not found, please check paths:"
    for ppl in plibs:
      print pfx,ppl
    return None

  notfound=1
  for pth in pincs:
    if (os.path.exists(pth+'/adfh/ADFH.h')):
      extraargs+=['-D__ADF_IN_SOURCES__']
      notfound=0
      break
    if (os.path.exists(pth+'/ADFH.h')):
      notfound=0
      break

  if notfound:
    print pfx+"***** warning ADFH.h not found, using pyCGNS own headers"
    extraargs+=['-U__ADF_IN_SOURCES__']

  libs=list(set(libs))
  return (cgnsversion,pincs,plibs,libs,extraargs,ifound,lfound)

# --------------------------------------------------------------------
def find_numpy(pincs,plibs,libs):
  import numpy
  vers=''
  extraargs=[]
  pdir=os.path.normpath(sys.prefix)
  xdir=os.path.normpath(sys.exec_prefix)
  pincs+=['%s/lib/python%s/site-packages/numpy/core/include'\
         %(xdir,sys.version[:3])]
  pincs+=['%s/lib/python%s/site-packages/numpy/core/include'\
         %(pdir,sys.version[:3])]
  pincs+=[numpy.get_include()]
  notfound=1      
  pincs=unique_but_keep_order(pincs)
  plibs=unique_but_keep_order(plibs)
  for pth in pincs:
    if (os.path.exists(pth+'/numpy/ndarrayobject.h')):
      fh=open(pth+'/numpy/ndarrayobject.h','r')
      fl=fh.readlines()
      fh.close()
      found=0
      for ifh in fl:
        if (ifh[:20] == "#define NPY_VERSION "):
          vers=ifh.split()[-1]
          found=1
      if found:
        pincs=[pth]
        notfound=0
        break
    if (os.path.exists(pth+'/numpy/_numpyconfig.h')):
      fh=open(pth+'/numpy/_numpyconfig.h','r')
      fl=fh.readlines()
      fh.close()
      found=0
      for ifh in fl:
        if (ifh[:24] == "#define NPY_ABI_VERSION "):
          vers=ifh.split()[-1]
          found=1
      if found:
        pincs=[pth]
        notfound=0
        break
  if notfound:
    print pfx,"ERROR: numpy headers not found, please check your paths"
    print pfx,pincs
    return None
  
  return (vers,pincs,plibs,libs,extraargs)

# --------------------------------------------------------------------
def touch(filename):
  now=time.time()
  os.utime(filename,(now,now))

# --- last line

