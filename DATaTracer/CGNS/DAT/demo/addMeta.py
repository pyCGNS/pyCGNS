#
#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# Add meta data skeletton in a file (uses ADF for efficiency)
# $Id: cgutils.py 0 2003-06-18 12:29:31Z mpoinot $
#
import CGNS
import NumArray
import os
import sys
import DAX.db.parseCGNS as px
import DAX.utils as ut
#
#
def addMeta(f,dct):
  a=CGNS.pyADF(f,CGNS.OLD,CGNS.NATIVE)
  px.checkMeta(a)
  px.changeMeta(a,dct)
  a.database_close()
#
if (__name__ == "__main__"):
  try:
    id=sys.argv[1]
    f="%s/%s.cgns"%(sys.argv[2],id)
    print "FILE: ",f
    dt=ut.getStat(f)[4]
    dct={id:{
     '.MetaData/CheckSum':         NumArray.array(ut.checksum(f),'c'),
     '.MetaData/GlobalCheckSum':   NumArray.array('000','c'),
     '.MetaData/CreationDate':     NumArray.array(dt,'c'),
     '.MetaData/ModificationDate': NumArray.array(dt,'c'),
     '.MetaData/Platform':         NumArray.array(os.uname()[1],'c'),
     '.MetaData/Owner':            NumArray.array('BS','c'),
     '.MetaData/Policy':           NumArray.array('PUBLIC','c'),
     '.MetaData/Version':          NumArray.array(0,'i'),
     '.MetaData/Release':          NumArray.array(0,'i'),
     '.MetaData/Change':           NumArray.array(1,'i'),
     '.MetaData/Title':            NumArray.array('No title set','c'),
     '.CHANCE/CaseNumber':         NumArray.array('XX-0','c'),
     '.CHANCE/Family':             NumArray.array('???','c'),
     '.CHANCE/Measurements':       NumArray.array('???','c'),
    },
    }   
    addMeta(f,dct[id])
    addMeta(f,{})    
  except IndexError:
    print "usage: addMeta.py <base-id> <file-path>"
