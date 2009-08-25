#!/usr/bin/env python
# pyDAX - CGNS files management package
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyDAX - $Id: setup.py 108 2003-11-13 15:29:58Z elsa $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
# 

from distutils.core import setup, Extension

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
                'DAX.demo',                
                'DAX.test'],
scripts      = ['DAX/tools/daxDB',
                'DAX/tools/daxQT',
                'DAX/tools/daxET'],
) # close setup

