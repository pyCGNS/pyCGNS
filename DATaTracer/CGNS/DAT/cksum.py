#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
# from:
# ONERA/DSNA/ELSA - poinot@onera.fr
# Updated 02/04/2003
#
# Computes the checksum in a deterministic way.
# Avoid all nodes with name [.checksum] which would contain the checksum
# -> NEVER FOLLOWS LINKS
#
# Usage:
#
# adfcheckum.py <filename>
#
# returns a fingerprint as a 32 characters hexa string
#
# Algorithm:
# - get first node, put children names in L
# - sort L using alphanumerical order
# - apply the nodechecksum (see below) to every L node in that order 
# - update md5 with this nodechecksum
#
# Nodechecksum:
# - get in that order
#   nodename+nodelabel+datatype+numberofdims+dims+data
#
# Notes:
# - Number of children, IDs, ADF versions, etc... are not taken into account
# - Links are not followed because checksum is per-file-basis
# - Data is completely read as a binary string
# - Uses pyCGNS, numpy, may lead to not portable fingerprint result
#
import CGNS
import numpy
import md5

# ----------------------------------------------------------------------
# checksums are made in a way it could be done by another software using
# same md5 and same strings
# - real strings are forced to 32 chars, padding with spaces, left justif.
#   using %-32.32s format
# - integers are %d format
# - actual node data is seen as an hexa string
def nodeChecksum(node,data,cks):
  cks.update("%-32.32s"%node['name'])          
  cks.update("%-32.32s"%node['label'])          
  cks.update("%-32.32s"%node['datatype'])       
  cks.update("%d"%len(node['dimensions']))
  for n in range(len(node['dimensions'])-1):
    cks.update("%d"%node['dimensions'][n])
  if (data != None): cks.update(data.tostring())  
    
# ----------------------------------------------------------------------
def parseAndUpdate(db,nodeid,cks):
  node=db.nodeAsDict(nodeid)
  dt=node['datatype']
  if (node['name'] != '.checksum'):
    if ((dt != "MT") and (dt != "LK")): data=db.read_all_data(nodeid)
    else:                               data=None
    nodeChecksum(node,data,cks)
  if (dt != "LK"):
    clist=list(node['children'])
    clist.sort()
    for c in clist:
      parseAndUpdate(db,db.get_node_id(nodeid,c),cks)
  
# ----------------------------------------------------------------------
def checksumtree(f,cks):
  parseAndUpdate(f,f.root(),cks)
  return cks
    
# ----------------------------------------------------------------------
usage="""usage: adfchecksum.py <filename>"""

if (__name__ == "__main__"):
  import sys
  import os
  try:
    filename=sys.argv[1]
  except IndexError:
    raise usage
  if (os.path.isfile(filename)):
      f=CGNS.pyADF(filename,CGNS.READ_ONLY,CGNS.NATIVE)
      cks=md5.new()
      print checksumtree(f,cks).hexdigest()
      f.database_close()
  else:
      raise usage
