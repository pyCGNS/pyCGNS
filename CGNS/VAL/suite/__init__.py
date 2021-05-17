#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
pass
# run the whole suite with:
#
#   python -c 'import CGNS.VAL.suite.run'
#
# To save all tests files in a CGNS/HDF files, set the shell variable
# CGNS_VAL_SAVE_FILES, create a directory and run the test in this directory.
#
#   mkdir files
#   cd files
#   export CGNS_VAL_SAVE_FILES=1
#   python -c 'import CGNS.VAL.suite.run'
#
# To run and compare with cgnscheck:
#
#   export CGNS_VAL_RUN_CGNSCHECK=1
#
# Files *.hdf are results or the CGNS/Python tree creation, *-fail-*
# trees may be unreadable.
# Files *.diag are CGNS.VAL output diagnostic
# Files *.cgnscheck are output files from cgnscheck
#
# Do not forget to remove files from one run to another...
#
