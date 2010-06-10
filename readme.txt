Python module information
=========================

The pyCGNS Python module is a collection of 7 modules around the
CGNS standard. 

 pyCGNS is released under the LGPL license
 See file COPYING in the root directory of this Python module source 
 tree for license information. 

The pyCGNS module now includes (former package names)

 VAL - Validater   (pyC5)   - XML grammar based validation of a CGNStree.py
 TRA - Translater  (pyCRAB) - Set of translators from/to various formats
 MAP - Mapper               - Load/save function SIDS/HDF5 from/to CGNStree.py
 DAT - DataTracer  (pyDAX)  - DBMS services for SIDS/HDF5 files
 WRA - Wrapper     (pyCGNS) - CGNS/MLL and CGNS/ADF python wrapping
 PAT - PatterMaker          - Full CGNS/SIDS patterns with CGNStree.py
 NAV - Navigater   (pyS7)   - CGNStree.py graphical browser 

Release notes
-------------
 Many changes in this v4 release, you can only use MAP, WRA, PAT and NAV.
 The other modules, VAL, TRA and DAT are present for archival/development
 purpose but you should NOT use them.

Changes
-------
 - cgnserrors changed to full exceptions at pyCGNS global level

Module dependancies
-------------------

The pyCGNS modules have dependancies with their brothers. The list below
gives you the required modules (or optional) for each of them.

 - MAP : None
 - PAT : MAP
 - WRA : PAT MAP
 - NAV : PAT MAP (WRA)

Install notes
-------------

NAV:
~~~~
The TkTreectrl module is required. You first need to install tktreectrl
(last version tested is tktreectrl-2.2.3) and TkinterTreectrl to map it
to Python (last version tested is TkinterTreectrl-0.8)

MAP:
~~~~
The CHLone librarie is required

WRA:
~~~~
CGNS/MLL and CGNS/ADF libraries are required
