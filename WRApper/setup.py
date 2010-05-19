#!/usr/bin/env python
# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 79 $ $Date: 2009-03-13 10:19:54 +0100 (Fri, 13 Mar 2009) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import os
from   distutils.core import setup, Extension
import distutils.util
import distutils.sysconfig

import getopt
import re
import glob
import string

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('WRA',['HDF5','MLL','numpy'])

# ---------------------------------------------------------------------------
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

lversion      = pyCGNSconfig.__vid__

extraargs=pyCGNSconfig.MLL_EXTRA_ARGS
include_dirs=mllpinc+hdfpinc
library_dirs=mllplib+hdfplib
optional_libs=mlllib+hdflib

configdict={}
setuputils.updateConfig("..","../build/lib/CGNS",configdict)
  
# --- add common stuff
include_dirs+=numpinc
include_dirs+=['CGNS/WRA/modadf']

# ***************************************************************************
# Setup script for the CGNS Python interface
ldescription  = "Python CFD General Notation System Interfaces"
lauthor       = "ONERA/DSNA/ELSA Marc Poinot"
lauthor_email = "poinot@onera.fr"
lurl          = "http://elsa.onera.fr/CGNS/releases"
llicense      = "Python"
lverbose      = 1
lpackages     = ['CGNS.WRA']
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
                        libraries    = optional_libs)
              ] # close extension modules

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()
                
setup (
  name         = 'pyCGNS.WRA',
  version      = lversion,
  description  = ldescription,
  author       = lauthor,
  author_email = lauthor_email,
  url          = lurl,
  license      = llicense,
  verbose      = lverbose,
  ext_modules  = lext_modules,
  packages     = lpackages,
  scripts      = lscripts,
  data_files   = ldata_files,

  cmdclass={'clean':setuputils.clean}

) # close setup

#print '### Leave the installation directory and test with:'
#print "python -c 'import CGNS.WRA;CGNS.WRA.test()'"
  
# --- last line
  
