#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------

run the whole suite with:

  python -c 'import CGNS.VAL.suite.run'

if the CGNS.PRO package is found, it is used instead of the SAMPLE grammar.

To save all tests files in a CGNS/HDF files, set the shell variable
CGNS_VAL_SAVE_FILES, create a directory and run the test in this directory.

  mkdir files
  cd files
  export CGNS_VAL_SAVE_FILES=1
  python -c 'import CGNS.VAL.suite.run'

Do not forget to remove files from one run to another...
