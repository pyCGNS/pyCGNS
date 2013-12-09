#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------

import CGNS.VAL.simplecheck 

def check(t,trace=False):
  chk=CGNS.VAL.simplecheck.compliant(t,userlist=['SIDS','elsA'])
  if trace:
      for d in chk[1]:
          print d
  return chk[0]

