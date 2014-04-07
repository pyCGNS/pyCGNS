#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
# 
from  distutils.core import setup, Extension
from  distutils.util import get_platform
import numpy
import glob
import os
from Cython.Distutils import build_ext

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('VAL')

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

incdirs=['%s/lib/python%s/site-packages/numpy/core/include'\
         %(os.path.normpath(sys.exec_prefix),sys.version[:3]),
         '.',
         'CGNS/APP/lib']
incdirs+=[numpy.get_include()]

extmods=[]
extmods+=[Extension("CGNS.VAL.grammars.CGNS_VAL_USER_SIDS_",
                    ["CGNS/VAL/grammars/CGNS_VAL_USER_SIDS_.pyx"],
                    include_dirs = incdirs,
                    extra_compile_args=[])]
extmods+=[Extension("CGNS.VAL.grammars.etablesids",
                    ["CGNS/VAL/grammars/etablesids.pyx"],
                    include_dirs = incdirs,
                    extra_compile_args=[])]
extmods+=[Extension("CGNS.VAL.grammars.valutils",
                    ["CGNS/VAL/grammars/valutils.pyx"],
                    include_dirs = incdirs,
                    extra_compile_args=[])]

cmdclassdict={'clean':setuputils.clean,'build_ext': build_ext}

# ---

setup (
name         = pyCGNSconfig.NAME,
version      = pyCGNSconfig.VERSION,
description  = pyCGNSconfig.DESCRIPTION,
author       = pyCGNSconfig.AUTHOR,
author_email = pyCGNSconfig.EMAIL,
license      = pyCGNSconfig.LICENSE,
packages     = ['CGNS.VAL',
                'CGNS.VAL.grammars',
                'CGNS.VAL.suite',
                'CGNS.VAL.suite.elsA',
                'CGNS.VAL.suite.SIDS',
                'CGNS.VAL.parse'],
scripts      = ['CGNS/VAL/CGNS.VAL'],
ext_modules  = extmods,
cmdclass     = cmdclassdict
) # close setup

