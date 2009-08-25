# ------------------------------------------------------------
# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyCCCCC - $Id: exceptions.py 22 2005-02-02 09:57:08Z  $
#
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
# ------------------------------------------------------------
import CCCCC.utils.stuff as ustf

xcount=0

class cgnsParserException(Exception):
  def __init__(self,id=0):
    global xcount
    if id: self.id=id
    else:  self.id=xcount
    ustf.ttt("Create exception code [%d]"%self.id)
  def __str__(self):
    return self.id

