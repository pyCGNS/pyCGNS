#!/usr/bin/env python
# pyDAX - CGNS files management package
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyDAX - $Id: setup.py 108 2003-11-13 15:29:58Z elsa $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
# 

from distutils.core import setup, Extension

# --- pyCGNSconfig search
import os
import sys
spath=sys.path[:]
sys.path=[os.getcwd(),'%s/..'%(os.getcwd())]
try:
  import pyCGNSconfig
except ImportError:
  print 'pyGCNS[ERROR]: PAT cannot find pyCGNSconfig.py file!'
  sys.exit(1)
sys.path=[os.getcwd(),'%s/..'%(os.getcwd())]+spath
import setuputils
setuputils.installConfigFiles([os.getcwd(),'%s/..'%(os.getcwd())])
sys.prefix=sys.exec_prefix
# ---

setup(
name         = "pyDAX",
version      = "1.0",
description  = "Data eXchage and Archival for CGNS files",
author       = "ONERA/DSNA/ELSA Marc Poinot",
author_email = "poinot@onera.fr",
url          = "-",
license      = "-",
verbose      = 1,

packages     = ['DAX',
                'DAX.db',
                'DAX.db.dbdrivers',                
                'DAX.demo'],
scripts      = ['DAX/tools/daxDB',
                'DAX/tools/daxQT',
                'DAX/tools/daxET'],
  cmdclass={'clean':setuputils.clean}
) # close setup

