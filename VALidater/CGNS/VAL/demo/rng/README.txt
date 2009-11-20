#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
# ------------------------------------------------------------
Grammar restrictions

 c5 -r g01.rng myfile.xml

g01 - Restricts MassUnits to Kilogram and Gram
g02 - Restricts Unstructured elements types

# ------------------------------------------------------------
Sub-grammar (check only a part of a CGNS tree)

g10 - Grammar for a Grid sub tree only
g11 - Grammar for a BC sub tree only