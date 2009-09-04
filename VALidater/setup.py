#!/usr/bin/env python
# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyC5 - $Id: setup.py 34 2005-02-16 14:38:32Z  $
#
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#

# Setup script for the CGNS Python interface
from distutils.core import setup, Extension
import distutils.sysconfig

import re
import sys
import glob
import getopt

# --- pyCGNSconfig search
import sys
sys.path+=['..']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('MAP')
# ---

dsubdir="share/CGNS"
dsubdir1="share/CGNS/SIDS/grammar"
dsubdir2="share/CGNS/Demo"
dsubdir3="share/CGNS/SIDS/examples"
dsubdir4="share/CGNS/Demo/cgns"
dsubdir5="share/CGNS/Demo/rng"
dsubdir6="share/CGNS/Demo/rnc"
dsubdir7="share/CGNS/USER/Eclipps"

ddir ="file://"+sys.prefix+"/"+dsubdir   # may be overwritten below

opts, args = getopt.getopt(sys.argv[1:],"","--install")
if (len(args)>1):
  for arg in args:
    if ((len(arg) > 10) and (arg[:9] == '--prefix=')):
      ddir="file://"+arg.split('=')[1]+"/"+dsubdir 
      print 'Change install dir to :', ddir

r1=re.compile(r"""
['"].*/cgns[.]rng["']
""",re.VERBOSE)

r2=re.compile(r"""
['"].*/sids-[^.]+[.]rng["']
""",re.VERBOSE)

r3=re.compile(r"""
file://TOBESETATINSTALLTIME
""",re.VERBOSE) 

import os
try:
  os.mkdir('build')
except OSError:
  pass

def substituteAbsolutePath(path,file,pattern,replace):
  print "substitute",replace,"in",file
  nfile=file+".tmp"
  fsrc=open(file,'r')
  fdst=open(nfile,'w+')
  for l in fsrc.readlines():
    nl=pattern.sub(replace,l)
    fdst.write(nl)
  fsrc.close()  
  fdst.close()
  os.rename(nfile,"build/"+os.path.split(file)[-1]) # change to distutil var

# files to translate before install
fdatalist=[
  'CCCCC/utils/filepaths.gen',
  'CCCCC/schema/sch/cgns.xsl',
  'CCCCC/schema/rng/sids.rng',
  'CCCCC/schema/rnc/sids.rnc',
  'CCCCC/schema/rng/cgns.rng',
  'CCCCC/schema/rnc/cgns.rnc'
  ]

# files to install
# SIDS
gfiles=[
  'build/cgns.rnc',
  'build/cgns.rng',
  'build/sids.rnc',
  'build/sids.rng',
  ]

gfiles+=glob.glob('CCCCC/schema/rnc/sids-*.rnc')
gfiles+=glob.glob('CCCCC/schema/rng/sids-*.rng')

demorngfiles=[]
demorngfiles+=glob.glob('CCCCC/demo/rng/*')

demorncfiles=[]
demorncfiles+=glob.glob('CCCCC/demo/rnc/*')

demousrfiles=[
  'build/c5semantic.sch',
  'build/c5syntax.rnc',
  'build/c5syntax.rng'
]

fdatalist+=demorngfiles
fdatalist+=demorncfiles

# non-SIDS (c5/xml system)
xfiles=[
  'build/cgns.xsl',  
  'CCCCC/schema/sch/skeleton1-6.xsl',
  'CCCCC/schema/sch/RNG2Schtron.xsl',
  'CCCCC/schema/sch/cgns.xsl',
  'CCCCC/schema/sch/cgns.xslt',
  'CCCCC/schema/pp/cgnsPP.xsl',
  'CCCCC/schema/sch/cgns.sch',
  ]

# SIDS-examples
explfiles=[]

explfiles+=glob.glob('CCCCC/demo/sids/*.xml')

# non-SIDS examples
demofiles=[
  'CCCCC/demo/README.txt',
  ]

demofiles+=glob.glob('CCCCC/demo/*.xml')

webfiles=[]
webfiles+=glob.glob('CCCCC/demo/cgns/*.xml')

# change path
if 0:
  import os
  for fdata in fdatalist:
    substituteAbsolutePath(ddir,fdata,r3,'%s'%(ddir))

#print fdatalist[0][:-4]+".py"
#os.rename("build/"+os.path.split(fdatalist[0])[-1],fdatalist[0][:-4]+".py")

sys.path.append('./CCCCC')
from __init__ import __vid__

setup (
name         = "CGNS.VAL",
version      = __vid__,
description  = "XML tools for CFD General Notation System",
author       = "ONERA/DSNA Poinot, Henaux",
author_email = "poinot@onera.fr,henaux@onera.fr",
url          = "http://elsa.onera.fr/CGNS/releases",
license      = "Python",
verbose      = 1,
packages     = ['CCCCC','CCCCC.demo',
                'CCCCC.gui','CCCCC.gui.Icons','CCCCC.utils','CCCCC.tools',
                'CCCCC.tools.rnc2rng',
                'CCCCC.parser'],
scripts      = ['CCCCC/tools/c5',
                'CCCCC/tools/crnc2rng',
                'CCCCC/tools/cgt',
                ],
data_files   = [ (dsubdir1,gfiles),       # SIDS grammar files
                 (dsubdir2,demofiles),    # non-SIDS examples files
                 (dsubdir3,explfiles),    # SIDS examples files
                 (dsubdir4,webfiles),     # CGNS web examples files
                 (dsubdir5,demorngfiles), # RNG restricted examples
                 (dsubdir6,demorncfiles), # RNC restricted examples
                 (dsubdir7,demousrfiles), # Full user restriction examples
                 (dsubdir,xfiles),        # non-SIDS xml files
               ],
  cmdclass={'clean':setuputils.clean}

) # close setup

