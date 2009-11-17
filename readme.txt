 ------------------------------------------------------------------------------

 pyCGNS v4.0 - The CGNS Python module 

 ------------------------------------------------------------------------------
 pyCGNS is released under the LGPL license
 See file COPYING in the root directory of this Python module source 
 tree for license information. 
 ------------------------------------------------------------------------------

 The pyCGNS module now includes (former package names):

  VAL - Validater   (pyC5)   - XML grammar based validation of a CGNStree.py
  TRA - Translater  (pyCRAB) - Set of translators from/to various formats
  MAP - Mapper               - Load/save function SIDS/HDF5 from/to CGNStree.py
  DAT - DataTracer  (pyDAX)  - DBMS services for SIDS/HDF5 files
  WRA - Wrapper     (pyCGNS) - CGNS/MLL and CGNS/ADF python wrapping
  PAT - PatterMaker          - Full CGNS/SIDS patterns with CGNStree.py
  NAV - Navigater   (pyS7)   - CGNStree.py graphical browser 

 ------------------------------------------------------------------------------
 RELEASE NOTES:

  Many changes in this v4 release, you can only use MAP, WRA, PAT and NAV.
  The other modules, VAL, TRA and DAT are present for archival/development
  purpose but you should NOT use them.

 ------------------------------------------------------------------------------
 CHANGES:

  - cgnserrors changed to full exceptions at pyCGNS global level

 ------------------------------------------------------------------------------
 DEPS:

  - PAT : MAP
  - WRA : PAT MAP
 ------------------------------------------------------------------------------
