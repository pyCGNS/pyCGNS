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

doc1="""
  pyCGNS installation setup -- pyCGNS v%s
  
  All setup options are unchanged, added options are:

"""

doc2="""
  Examples:

  python setup.py build --includes=/usr/local/includes:/home/tools/include
"""

pr=argparse.ArgumentParser(description=doc1,epilog=doc2,
                           formatter_class=argparse.RawDescriptionHelpFormatter,
                           usage='python %(prog)s [options] file1 file2 ...')

pr.add_argument("-I","--includes",dest="incs",
                help='list of paths for include search ( : separated)')
pr.add_argument("-L","--libraries",dest="libs",
                help='list of paths for libraries search ( : separated)')
pr.add_argument("-U","--update",action='store_true',
                help='update version (dev only)')

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

incs=[]
libs=[]

args,unknown=pr.parse_known_args()

if (args.incs!=None):
  incs=[os.path.expanduser(path) for path in args.incs.split(':')]
if (args.libs!=None):
  libs=[os.path.expanduser(path) for path in args.libs.split(':')]

try:
  (CONFIG,status)=setuputils.search(incs,libs)
except setuputils.ConfigException, val:
  print 'Cannot build pyCGNS without:',val
  sys.exit(1)

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

def line(m):
   print "###","-"*70
   print "### %s"%m

# -------------------------------------------------------------------------
if APP:
  slist=['cg_grep','cg_list','cg_link',
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

# -------------------------------------------------------------------------  
if (MAP and CONFIG.HAS_CHLONE):
   ALL_PACKAGES+=['CGNS.MAP','CGNS.MAP.test']

# -------------------------------------------------------------------------  
if VAL:
  ALL_PACKAGES+=['CGNS.VAL',
                 'CGNS.VAL.grammars',
                 'CGNS.VAL.suite',
                 'CGNS.VAL.suite.elsA',
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

# -------------------------------------------------------------------------  
if PAT:
  #if CONFIG.HAS_CYTHON:
  #  ALL_EXTENSIONS+=[ Extension('CGNS.PAT.cgnsutils',
  #                              ['CGNS/PAT/cgnsutils.pyx'],
  #                              include_dirs = CONFIG.INCLUDE_DIRS) ]
  ALL_PACKAGES+=['CGNS.PAT',
                 'CGNS.PAT.SIDS',
                 'CGNS.PAT.elsA',
                 'CGNS.PAT.test']

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

# -------------------------------------------------------------------------  
if (NAV and CONFIG.HAS_PYSIDE):
  cui=CONFIG.COM_UIC
  crc=CONFIG.COM_RCC
  ccy=CONFIG.COM_CYTHON

  fakefile="./CGNS/NAV/fake.pxi"
  #setuputils.touch(fakefile)

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
         or os.path.getmtime(g[0])>os.path.getmtime(g[1])): modgenlist+=[m]
                  
  modextlist+=[Extension("CGNS.NAV.temputils",["CGNS/NAV/temputils.pyx",
                                               fakefile],
                         include_dirs = CONFIG.NUMPY_PATH_INCLUDES,
                         library_dirs = CONFIG.NUMPY_PATH_LIBRARIES,
                         libraries    = CONFIG.NUMPY_LINK_LIBRARIES,
                       )]

  for m in modgenlist:
      print '### pyCGNS: Generate from updated GUI templates: ',m
      com="(%s -o CGNS/NAV/G/%s.pyx CGNS/NAV/T/%s.ui;(cd CGNS/NAV/G;%s -a %s.pyx))2>/dev/null"%(cui,m,m,ccy,m)
      print com
      os.system(com)
         
  if (os.path.getmtime('CGNS/NAV/R/Res.qrc')>os.path.getmtime('CGNS/NAV/Res_rc.py')):
      print '### pyCGNS: Generate from updated GUI Ressources'
      com="(%s -o CGNS/NAV/Res_rc.py CGNS/NAV/R/Res.qrc)2>/dev/null"%(crc)
      print com
      os.system(com)

  ALL_PACKAGES+=['CGNS.NAV','CGNS.NAV.test']
  ALL_SCRIPTS+=['CGNS/NAV/CGNS.NAV']
  ALL_EXTENSIONS+=modextlist

setuputils.installConfigFiles(CONFIG.PRODUCTION_DIR)

#  -------------------------------------------------------------------------
if (CONFIG.HAS_CYTHON):
  from Cython.Distutils import build_ext
  cmd={'clean':setuputils.clean,'build_ext':build_ext}
else:
  cmd={'clean':setuputils.clean}

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

