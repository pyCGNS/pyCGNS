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

os.system('hg parents --template="{rev}\n" > ./lib/revision.tmp')
setuputils.updateVersionInFile('./lib/pyCGNSconfig_default.py')

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

if (args.incs!=None): incs=args.incs.split(':')
if (args.libs!=None): libs=args.libs.split(':')

try:
  CONFIG=setuputils.search(incs,libs)
except setuputils.ConfigException, val:
  print 'Cannot build pyCGNS without:',val
  sys.exit(1)

def line(m):
   print "###","-"*70
   print "### %s"%m

# -------------------------------------------------------------------------
if APP:
  line('APP')

  incdirs=['%s/lib/python%s/site-packages/numpy/core/include'\
           %(os.path.normpath(sys.exec_prefix),sys.version[:3]),
           '.',
           'CGNS/APP/lib']
  incdirs+=[numpy.get_include()]

  slist=['cg_grep','cg_list','cg_link',
         'cg_gather','cg_scatter',
         'cg_scan','cg_look']

  ALL_SCRIPTS+=['CGNS/APP/tools/%s'%f for f in slist]

  ALL_EXTENSIONS+=[Extension("CGNS.APP.lib.arrayutils",
                             ["CGNS/APP/lib/arrayutils.pyx",
                              "CGNS/APP/lib/hashutils.c"],
                             include_dirs = incdirs,
                             extra_compile_args=[])]
  ALL_PACKAGES+=['CGNS.APP',
                 'CGNS.APP.lib',
                 'CGNS.APP.tools',
                 'CGNS.APP.examples',
                 'CGNS.APP.misc'],

# -------------------------------------------------------------------------  
if (MAP and CONFIG.HAS_CHLONE):
   line('MAP')
   ALL_PACKAGES+=['CGNS.MAP','CGNS.MAP.test'],

# -------------------------------------------------------------------------  
if VAL:
  line('VAL')

  incdirs=['%s/lib/python%s/site-packages/numpy/core/include'\
           %(os.path.normpath(sys.exec_prefix),sys.version[:3]),
           '.',
           'CGNS/APP/lib']
  incdirs+=[numpy.get_include()]

  ALL_PACKAGES+=['CGNS.VAL',
                 'CGNS.VAL.grammars',
                 'CGNS.VAL.suite',
                 'CGNS.VAL.suite.elsA',
                 'CGNS.VAL.suite.SIDS',
                 'CGNS.VAL.parse']
  ALL_SCRIPTS+=['CGNS/VAL/CGNS.VAL']

  ALL_EXTENSIONS+=[Extension("CGNS.VAL.grammars.CGNS_VAL_USER_SIDS_",
                             ["CGNS/VAL/grammars/CGNS_VAL_USER_SIDS_.pyx"],
                             include_dirs = incdirs,
                             extra_compile_args=[])]
  ALL_EXTENSIONS+=[Extension("CGNS.VAL.grammars.etablesids",
                             ["CGNS/VAL/grammars/etablesids.pyx"],
                             include_dirs = incdirs,
                             extra_compile_args=[])]
  ALL_EXTENSIONS+=[Extension("CGNS.VAL.grammars.valutils",
                             ["CGNS/VAL/grammars/valutils.pyx"],
                             include_dirs = incdirs,
                             extra_compile_args=[])]

# -------------------------------------------------------------------------  
if PAT:
  line('PAT')

  if CONFIG.HAS_CYTHON:
    ALL_EXTENSIONS+=[ Extension('CGNS.PAT.cgnsutils',
                                ['CGNS/PAT/cgnsutils.pyx'],
                                include_dirs = CONFIG.NUMPY_PATH_INCLUDES) ]
  ALL_PACKAGES+=['CGNS.PAT','CGNS.PAT.SIDS','CGNS.PAT.test']

# -------------------------------------------------------------------------  
if (WRA and CONFIG.HAS_MLL):
  line('WRA')
  # -----------------------------------------------------------------------
  # --- config values
  hdfplib=pyCGNSconfig.HDF5_PATH_LIBRARIES
  hdflib=pyCGNSconfig.HDF5_LINK_LIBRARIES
  hdfpinc=pyCGNSconfig.HDF5_PATH_INCLUDES
  hdfversion=pyCGNSconfig.HDF5_VERSION
  mllplib=pyCGNSconfig.MLL_PATH_LIBRARIES
  mlllib=pyCGNSconfig.MLL_LINK_LIBRARIES
  mllpinc=pyCGNSconfig.MLL_PATH_INCLUDES
  mllversion=pyCGNSconfig.MLL_VERSION
  numpinc=pyCGNSconfig.NUMPY_PATH_INCLUDES

  # --- default values
  mll=True
  hdf=False
  cgnslib=""
  hdfversion=cgnsversion='unknown'

  lname    = "CGNS.WRA"

  extraargs=pyCGNSconfig.MLL_EXTRA_ARGS
  include_dirs=mllpinc+hdfpinc+pyCGNSconfig.INCLUDE_DIRS
  library_dirs=mllplib+hdfplib+pyCGNSconfig.LIBRARY_DIRS
  optional_libs=mlllib+hdflib

  configdict={}
  #setuputils.updateConfig("..","../build/lib/CGNS",configdict)
    
  # --- add common stuff
  include_dirs+=numpinc
  include_dirs+=['WRA/modadf']

  # ************************************************************************
  # Setup script for the CGNS Python interface
  lverbose      = 1
  lpackages     = ['CGNS.WRA','CGNS.WRA.test']
  lscripts      = []
  ldata_files   = []
  lext_modules  = [
                # You must let adfmodule into the midlevel shared library
                # ADF has some static variables, and changing module .so
                # will let the values separate, one in each .so
                # Thus, adf module has to be duplicated and the calls to
                # adf through midlevel should be clearly scoped in the
                # python code
                Extension('CGNS.WRA._mllmodule', 
                sources=['CGNS/WRA/modadf/adfmodule.c',
                         'CGNS/WRA/modmll/cgnsmodule.c',
                         'CGNS/WRA/modmll/cgnsdict.c'],
                          include_dirs = include_dirs,
                          library_dirs = library_dirs,
                          libraries    = optional_libs,
                          extra_compile_args=extraargs),
                Extension('CGNS.WRA._adfmodule', 
                sources=['CGNS/WRA/modadf/adfmodule.c'],
                          include_dirs = include_dirs,
                          library_dirs = library_dirs,
                          libraries    = optional_libs),
                ] # close extension modules

  if CONFIG.HAS_CYTHON:
    lext_modules+=[Extension('CGNS.WRA.mll',['CGNS/WRA/mll.pyx',
                                             'CGNS/WRA/mll_utils.c'],
                             include_dirs = include_dirs+['WRA'],
                             library_dirs = library_dirs,
                             libraries    = optional_libs)]
             
  setuputils.installConfigFiles()
                  
  setup (
  name         = pyCGNSconfig.NAME,
  version      = pyCGNSconfig.VERSION,
  description  = pyCGNSconfig.DESCRIPTION,
  author       = pyCGNSconfig.AUTHOR,
  author_email = pyCGNSconfig.EMAIL,
  license      = pyCGNSconfig.LICENSE,
  #  verbose      = lverbose,
  ext_modules  = lext_modules,
  packages     = lpackages,
  scripts      = lscripts,
  data_files   = ldata_files,
  )   

# -------------------------------------------------------------------------  
if DAT:
  line('DAT')

  ALL_PACKAGES+=['CGNS.DAT',
                 'CGNS.DAT.db',
                 'CGNS.DAT.db.dbdrivers',                
                 'CGNS.DAT.demo']
  ALL_SCRIPTS+=['CGNS/DAT/tools/CGNS.DAT',
                'CGNS/DAT/tools/daxQT',
                'CGNS/DAT/tools/CGNS.DAT.create']

# -------------------------------------------------------------------------  
if (NAV and CONFIG.HAS_PYSIDE):
  line('NAV')

  cui=CONFIG.COM_UIC
  crc=CONFIG.COM_RCC
  ccy=CONFIG.COM_CYTHON

  if installprocess:
    from optparse import OptionParser
    from distutils.core import setup, Extension
    import re

    sys.path.append('.')
    from pyCGNSconfig import version as __vid__
    from optparse import OptionParser,OptionError

    fakefile="./NAV/fake.pxi"

    parser = OptionParser()
    parser.add_option("--force",dest="forcerebuild",action="store_true")
    parser.add_option("--prefix",dest="prefix")
    parser.add_option("--dist",dest="dist")
    parser.add_option("--compiler",dest="compiler")
    parser.add_option("--build-base",dest="build-base")
    parser.add_option("--format",dest="format")
    try:
      (options, args) = parser.parse_args(sys.argv)
      if (options.forcerebuild):
        setuputils.touch(fakefile)
    except OptionError: pass

    setuputils.installConfigFiles()
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
                             include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                             library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                             libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
                             )]
    for m in modnamelist:
       modextlist+=[Extension("CGNS.NAV.%s"%m, ["CGNS/NAV/G/%s.pyx"%m,
                                                fakefile],
                              include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                              library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                              libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
                              )]
       g=("CGNS/NAV/T/%s.ui"%m,"CGNS/NAV/G/%s.pyx"%m)
       if (not os.path.exists(g[1])
           or os.path.getmtime(g[0])>os.path.getmtime(g[1])): modgenlist+=[m]
                    
    modextlist+=[Extension("CGNS.NAV.temputils",["CGNS/NAV/temputils.pyx",
                                                 fakefile],
                           include_dirs = pyCGNSconfig.NUMPY_PATH_INCLUDES,
                           library_dirs = pyCGNSconfig.NUMPY_PATH_LIBRARIES,
                           libraries    = pyCGNSconfig.NUMPY_LINK_LIBRARIES,
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
  else:
    modextlist=[]

  ALL_PACKAGES+=['CGNS.NAV']
  ALL_SCRIPTS+=['CGNS/NAV/CGNS.NAV']
  ALL_EXTENSIONS+=modextlist

#  -------------------------------------------------------------------------  

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
cmdclass     = {'clean':setuputils.clean,'build_ext':build_ext}
)  

#  -------------------------------------------------------------------------  

# --- last line

