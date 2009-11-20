#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
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

