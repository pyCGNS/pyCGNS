#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import os
import sys
import string
import argparse
import glob
import re

from distutils.core import setup, Extension
from distutils.util import get_platform
from distutils      import sysconfig

# --- get overall configuration
sys.path=['./lib']+sys.path
import setuputils

def line(m=""):
   print "#","-"*70
   if (m): print "# ----- %s"%m

line('pyCGNS v%d.%d install'%(setuputils.MAJORVERSION,setuputils.MINORVERSION))
line()

doc1="""
  pyCGNS installation setup 
  - Usual python setup options are unchanged (build, install, help...)
  - The recommanded way to build is to set your shell environment with
    your own PATH, LD_LIBRARY_PATH and PYTHONPATH so that the setup would
    find all expected ressources itself.
  - You can either use the command line args as described below or edit
    the setup_userConfig.py with the values you want.

  All packages are installed if all expected dependancies are found.
  See doc for more installation details and depend:
  http://pycgns.sourceforge.net/install.html

"""

doc2="""
  Examples:

  1. The best way is to let setup find out required stuff, build and
     install. This would need write access to python installation dirs:

  python setup.py install

  2. You can build and install in two separate commands

  python setup.py build
  python setup.py install

  3. Specific paths would be used prior to any automatic detection:

  python setup.py build --includes=/usr/local/includes:/home/tools/include

  4. Installation to a local directory (usual setup pattern)

  python setup.py install --prefix=/home/myself/install

"""

pr=argparse.ArgumentParser(description=doc1,epilog=doc2,
                           formatter_class=argparse.RawDescriptionHelpFormatter,
                           usage='python %(prog)s [options] file1 file2 ...')

pr.add_argument("-I","--includes",dest="incs",
                help='list of paths for include search ( : separated), order is significant and is kept unchanged')
pr.add_argument("-L","--libraries",dest="libs",
                help='list of paths for libraries search ( : separated), order is significant and is kept unchanged')
pr.add_argument("-U","--update",action='store_true',
                help='update version (dev only)')
pr.add_argument("-g","--generate",action='store_true',
                help='force Qt/creator .pyx files to be regenerated')

try:
  os.makedirs('./build/lib/CGNS')
except OSError:
  pass

APP=True
MAP=True
WRA=True
MAP=True
PAT=True
VAL=True
DAT=True
NAV=True

ALL_PACKAGES=[]
ALL_SCRIPTS=[]
ALL_EXTENSIONS=[]

modules=""

incs=[]
libs=[]

args,unknown=pr.parse_known_args()

if (args.incs!=None):
  incs=[os.path.expanduser(path) for path in args.incs.split(os.path.pathsep)]
if (args.libs!=None):
  libs=[os.path.expanduser(path) for path in args.libs.split(os.path.pathsep)]

try:
  (CONFIG,status)=setuputils.search(incs,libs)
except setuputils.ConfigException, val:
  print 'Cannot build pyCGNS without:',val
  sys.exit(1)

print setuputils.pfx+'-'*65

new_args=[]
for arg in sys.argv:
  if ( not ('-I=' in arg or '--includes=' in arg) and
       not ('-U' in arg or '--update' in arg) and
       not ('-L=' in arg or '--libraries=' in arg)):
    new_args+=[arg]
sys.argv=new_args

if (args.update):
  os.system('hg parents --template="{rev}\n" > %s/revision.tmp'\
            %CONFIG.PRODUCTION_DIR)
  setuputils.updateVersionInFile('./lib/pyCGNSconfig_default.py',
                                 CONFIG.PRODUCTION_DIR)

# -------------------------------------------------------------------------
if APP:
  slist=['cg_grep','cg_list','cg_link','cg_iges','cg_diff','cg_checksum',
         'cg_gather','cg_scatter',
         'cg_scan','cg_look']

  ALL_SCRIPTS+=['CGNS/APP/tools/%s'%f for f in slist]

  ALL_EXTENSIONS+=[Extension("CGNS.APP.lib.arrayutils",
                             ["CGNS/APP/lib/arrayutils.pyx",
                              "CGNS/APP/lib/hashutils.c"],
                             include_dirs = CONFIG.INCLUDE_DIRS,
                             extra_compile_args=[])]
  ALL_PACKAGES+=['CGNS.APP',
                 'CGNS.APP.lib',
                 'CGNS.APP.tools',
                 'CGNS.APP.examples',
                 'CGNS.APP.misc',
                 'CGNS.APP.test']
  modules+="\n# APP   add  build"
else:
  modules+="\n# APP   skip build *"

# -------------------------------------------------------------------------  
if (MAP and CONFIG.HAS_CHLONE):
  ALL_PACKAGES+=['CGNS.MAP','CGNS.MAP.test']
  modules+="\n# MAP   add  build"
else:
  modules+="\n# MAP   skip build *"

# -------------------------------------------------------------------------  
if VAL:
  ALL_PACKAGES+=['CGNS.VAL',
                 'CGNS.VAL.grammars',
                 'CGNS.VAL.suite',
                 'CGNS.VAL.suite.SIDS',
                 'CGNS.VAL.parse',
                 'CGNS.VAL.test']
  ALL_SCRIPTS+=['CGNS/VAL/CGNS.VAL']

  ALL_EXTENSIONS+=[Extension("CGNS.VAL.grammars.CGNS_VAL_USER_SIDS_",
                             ["CGNS/VAL/grammars/CGNS_VAL_USER_SIDS_.pyx"],
                             include_dirs = CONFIG.INCLUDE_DIRS,
                             extra_compile_args=[])]
  ALL_EXTENSIONS+=[Extension("CGNS.VAL.grammars.etablesids",
                             ["CGNS/VAL/grammars/etablesids.pyx"],
                             include_dirs = CONFIG.INCLUDE_DIRS,
                             extra_compile_args=[])]
  ALL_EXTENSIONS+=[Extension("CGNS.VAL.grammars.valutils",
                             ["CGNS/VAL/grammars/valutils.pyx"],
                             include_dirs = CONFIG.INCLUDE_DIRS,
                             extra_compile_args=[])]

  modules+="\n# VAL   add  build"
else:
  modules+="\n# VAL   skip build *"

# -------------------------------------------------------------------------  
if PAT:
  #if CONFIG.HAS_CYTHON:
  #  ALL_EXTENSIONS+=[ Extension('CGNS.PAT.cgnsutils',
  #                              ['CGNS/PAT/cgnsutils.pyx'],
  #                              include_dirs = CONFIG.INCLUDE_DIRS) ]
  ALL_PACKAGES+=['CGNS.PAT',
                 'CGNS.PAT.SIDS',
                 'CGNS.PAT.test']
  modules+="\n# PAT   add  build"
else:
  modules+="\n# PAT   skip build *"

# -------------------------------------------------------------------------  
if (WRA and CONFIG.HAS_MLL and CONFIG.HAS_CYTHON_2PLUS):

  # --- config values
  hdfplib=CONFIG.HDF5_PATH_LIBRARIES
  hdflib=CONFIG.HDF5_LINK_LIBRARIES
  hdfpinc=CONFIG.HDF5_PATH_INCLUDES
  hdfversion=CONFIG.HDF5_VERSION
  mllplib=CONFIG.MLL_PATH_LIBRARIES
  mlllib=CONFIG.MLL_LINK_LIBRARIES
  mllpinc=CONFIG.MLL_PATH_INCLUDES
  mllversion=CONFIG.MLL_VERSION
  numpinc=CONFIG.NUMPY_PATH_INCLUDES

  lname    = "CGNS.WRA"

  extraargs=CONFIG.MLL_EXTRA_ARGS
  include_dirs=CONFIG.INCLUDE_DIRS+['WRA/modadf']
  library_dirs=CONFIG.LIBRARY_DIRS
  optional_libs=mlllib+hdflib

  ALL_PACKAGES+=['CGNS.WRA','CGNS.WRA.test']

  if CONFIG.HAS_CYTHON:
    ALL_EXTENSIONS+=[Extension('CGNS.WRA.mll',['CGNS/WRA/mll.pyx',
                                               'CGNS/WRA/mll_utils.c'],
                               include_dirs = include_dirs+['WRA'],
                               library_dirs = library_dirs,
                               libraries    = optional_libs)]

  modules+="\n# WRA   add  build"
else:
  modules+="\n# WRA   skip build *"

# -------------------------------------------------------------------------  
if DAT:
  ALL_PACKAGES+=['CGNS.DAT',
                 'CGNS.DAT.db',
                 'CGNS.DAT.db.dbdrivers',                
                 'CGNS.DAT.demo',
                 'CGNS.DAT.test']
  ALL_SCRIPTS+=['CGNS/DAT/tools/CGNS.DAT',
                'CGNS/DAT/tools/daxQT',
                'CGNS/DAT/tools/CGNS.DAT.create']
  modules+="\n# DAT   add  build"
else:
  modules+="\n# DAT   skip build *"

# -------------------------------------------------------------------------  
if (NAV and CONFIG.HAS_PYQT4):
  cui=CONFIG.COM_UIC
  crc=CONFIG.COM_RCC
  ccy=CONFIG.COM_CYTHON

  fakefile="./CGNS/NAV/fake.pxi"
  if (args.generate): setuputils.touch(fakefile)

  modnamelist=[
      'Q7TreeWindow',
      'Q7DiffWindow',
      'Q7MergeWindow',
      'Q7ControlWindow',
      'Q7OptionsWindow',
      'Q7FormWindow',
      'Q7FileWindow',
      'Q7QueryWindow',
      'Q7SelectionWindow',
      'Q7InfoWindow',
      'Q7DiagWindow',
      'Q7LinkWindow',
      'Q7HelpWindow',
      'Q7ToolsWindow',
      'Q7PatternWindow',
      'Q7AnimationWindow',
      'Q7MessageWindow',
      'Q7LogWindow',
      ]
  if (CONFIG.HAS_VTK): modnamelist+=['Q7VTKWindow']
  modgenlist=[]
  modextlist=[]
  mfile_list=['mtree','mquery','mcontrol','mtable','mpattern',
              'diff','mdifftreeview','merge','mmergetreeview']
  if (CONFIG.HAS_VTK): mfile_list+=['mparser']
    
  for mfile in mfile_list:
     modextlist+=[Extension("CGNS.NAV.%s"%mfile,["CGNS/NAV/%s.pyx"%mfile,
                                                 fakefile],
                           include_dirs = CONFIG.NUMPY_PATH_INCLUDES,
                           library_dirs = CONFIG.NUMPY_PATH_LIBRARIES,
                           libraries    = CONFIG.NUMPY_LINK_LIBRARIES,
                           )]
  for m in modnamelist:
     modextlist+=[Extension("CGNS.NAV.%s"%m, ["CGNS/NAV/G/%s.pyx"%m,
                                              fakefile],
                            include_dirs = CONFIG.NUMPY_PATH_INCLUDES,
                            library_dirs = CONFIG.NUMPY_PATH_LIBRARIES,
                            libraries    = CONFIG.NUMPY_LINK_LIBRARIES,
                            )]
     g=("CGNS/NAV/T/%s.ui"%m,"CGNS/NAV/G/%s.pyx"%m)
     if (not os.path.exists(g[1])
         or args.generate
         or os.path.getmtime(g[0])>os.path.getmtime(g[1])): modgenlist+=[m]
                  
  modextlist+=[Extension("CGNS.NAV.temputils",["CGNS/NAV/temputils.pyx",
                                               fakefile],
                         include_dirs = CONFIG.NUMPY_PATH_INCLUDES,
                         library_dirs = CONFIG.NUMPY_PATH_LIBRARIES,
                         libraries    = CONFIG.NUMPY_LINK_LIBRARIES,
                       )]

  for m in modgenlist:
      print '# Generate from updated Qt templates  (%s): %s'%(cui,m)
      com="(%s -o CGNS/NAV/G/%s.pyx CGNS/NAV/T/%s.ui;(cd CGNS/NAV/G;%s -a %s.pyx))2>/dev/null"%(cui,m,m,ccy,m)
      #print com
      os.system(com)

  opg=os.path.getmtime
  if (args.generate or opg('CGNS/NAV/R/Res.qrc')>opg('CGNS/NAV/Res_rc.py')):
      print '# Generate from updated Qt Ressources (%s): Res_rc.py'%(crc)
      com="(%s -o CGNS/NAV/Res_rc.py CGNS/NAV/R/Res.qrc)2>/dev/null"%(crc)
      os.system(com)

  ALL_PACKAGES+=['CGNS.NAV','CGNS.NAV.test']
  ALL_EXTENSIONS+=modextlist

  modules+="\n# NAV   add  build"
else:
  modules+="\n# NAV   skip build *"

setuputils.installConfigFiles(CONFIG.PRODUCTION_DIR)

#  -------------------------------------------------------------------------
if (CONFIG.HAS_CYTHON):
  from Cython.Distutils import build_ext
  cmd={'clean':setuputils.clean,'build_ext':build_ext}
else:
  cmd={'clean':setuputils.clean}

print "#"+modules
print "#\n# Running build now...\n#"

# -------------------------------------------------------------------------  
setup (
name         = CONFIG.NAME,
version      = CONFIG.VERSION,
description  = CONFIG.DESCRIPTION,
author       = CONFIG.AUTHOR,
author_email = CONFIG.EMAIL,
license      = CONFIG.LICENSE,
packages     = ALL_PACKAGES,
scripts      = ALL_SCRIPTS,
ext_modules  = ALL_EXTENSIONS,
cmdclass     = cmd
)  
# -------------------------------------------------------------------------  
# --- last line

