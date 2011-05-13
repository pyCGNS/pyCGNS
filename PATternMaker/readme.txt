.. -------------------------------------------------------------------------
.. pyCGNS.PAT - CFD General Notation System - 
.. See license.txt file in the root directory of this Python module source  
.. -------------------------------------------------------------------------

CGNS.PAT
========

The module to create and manipulate SIDS/Python trees.
PAT has a *cgnslib* module with functions to create SIDS/Python
compliant data structures.
PAT defines all the CGNS types, names, enumerates or any other CGNS keyword.

SIDS patterns
-------------
This module contains all the CGNS/SIDS structures using CGNS.PAT as API.

The Pythonish CGNS lib
----------------------

The so-called `CGNS lib` or `MLL` or `Mid-level` library, is set of functions
for used to read/write/modify a set of nodes matching a CGNS/SIDS type.
The Pythonish flavour of this library declares a set of functions with more
or less the same interface but with Python values.

Utilities
---------

The `CGNS.PAT.cgnsutils` has a large set of utility functions.

.. automodule:: CGNS.PAT.cgnsutils
   :members:

CGNS Keywords
-------------
Instead of generating a new doc from a file, the file itself is included here.
The purpose of `cgnskeywords.py` is to declare all constants as Python 
variables. This leads to several advantages:

 * You cannot make a typo on a name. For example, if you use 
   "ZoneGridConnectivity" as a plain string you may mistype it as
   "Zonegridconnectivity" or "ZoneGridConectivity" and this may silently
   produce a bad CGNS tree.

 * You can handle enumerate as lists. For example you have lists for 
   units: MassUnits_l, LengthUnits_l, AllDimensionalUnits_l, AllUnits_l

 * You can identify what is a CGNS reserved or recommended name or not.

.. literalinclude:: CGNS/PAT/cgnskeywords.py
   
CGNS Types
----------
.. USE THIS COMMAND FOR FILE UPDATE
.. python -c 'import CGNS.APP.sids.checktypes as C;C.generateSphinx("PATternMaker/cgnstypes.txt")'

.. toctree::
   
   cgnstypes

CGNS.PAT.cgnserrors
---------------------

.. ### autodata:: CGNS.PAT.cgnserrors

-----

.. _pat_index:

PAT Index
---------

* :ref:`genindex`

.. -------------------------------------------------------------------------
