#!/usr/bin/env python
# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyC5 - $Id: setup.py 34 2005-02-16 14:38:32Z  $
#
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import os

# Setup script for the CGNS Python interface
from distutils.core import setup, Extension
import distutils.sysconfig

import re
import sys
import glob
import getopt

# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('VAL')
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
  'CGNS/VAL/utils/filepaths.gen',
  'CGNS/VAL/schema/sch/cgns.xsl',
  'CGNS/VAL/schema/rng/sids.rng',
  'CGNS/VAL/schema/rnc/sids.rnc',
  'CGNS/VAL/schema/rng/cgns.rng',
  'CGNS/VAL/schema/rnc/cgns.rnc'
  ]

# files to install
# SIDS
gfiles=[
  'build/cgns.rnc',
  'build/cgns.rng',
  'build/sids.rnc',
  'build/sids.rng',
  ]

gfiles+=glob.glob('CGNS/VAL/schema/rnc/sids-*.rnc')
gfiles+=glob.glob('CGNS/VAL/schema/rng/sids-*.rng')

demorngfiles=[]
demorngfiles+=glob.glob('CGNS/VAL/demo/rng/*')

demorncfiles=[]
demorncfiles+=glob.glob('CGNS/VAL/demo/rnc/*')

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
  'CGNS/VAL/schema/sch/skeleton1-6.xsl',
  'CGNS/VAL/schema/sch/RNG2Schtron.xsl',
  'CGNS/VAL/schema/sch/cgns.xsl',
  'CGNS/VAL/schema/sch/cgns.xslt',
  'CGNS/VAL/schema/pp/cgnsPP.xsl',
  'CGNS/VAL/schema/sch/cgns.sch',
  ]

# SIDS-examples
explfiles=[]

explfiles+=glob.glob('CGNS/VAL/demo/sids/*.xml')

# non-SIDS examples
demofiles=[
  'CGNS/VAL/demo/README.txt',
  ]

demofiles+=glob.glob('CGNS/VAL/demo/*.xml')

webfiles=[]
webfiles+=glob.glob('CGNS/VAL/demo/cgns/*.xml')

# change path
if 0:
  import os
  for fdata in fdatalist:
    substituteAbsolutePath(ddir,fdata,r3,'%s'%(ddir))

#print fdatalist[0][:-4]+".py"
#os.rename("build/"+os.path.split(fdatalist[0])[-1],fdatalist[0][:-4]+".py")

if (not os.path.exists("build")): os.system("ln -sf ../build build")
setuputils.installConfigFiles()

setup (
name         = "CGNS.VAL",
version      = pyCGNSconfig.VAL_VERSION,
description  = "pyCGNS VALidater - SIDS verification tools",
author       = "marc Poinot, elise Henaux",
author_email = "poinot@onera.fr,henaux@onera.fr",
license      = "LGPL 2",
verbose      = 1,
packages     = ['CGNS.VAL','CGNS.VAL.demo',
                'CGNS.VAL.gui','CGNS.VAL.gui.Icons','CGNS.VAL.utils','CGNS.VAL.tools',
                'CGNS.VAL.tools.rnc2rng',
                'CGNS.VAL.parser'],
scripts      = ['CGNS/VAL/tools/c5',
                'CGNS/VAL/tools/crnc2rng',
                'CGNS/VAL/tools/cgt',
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

