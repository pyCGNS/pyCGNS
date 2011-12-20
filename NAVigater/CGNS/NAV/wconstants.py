#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.PAT.cgnskeywords
import CGNS.PAT.cgnstypes

reservedNames=[n[:-2] for n in CGNS.PAT.cgnskeywords.cgnsnames]
reservedTypes=[n[:-1] for n in CGNS.PAT.cgnskeywords.cgnstypes]

FixedFontTable='Courier'


