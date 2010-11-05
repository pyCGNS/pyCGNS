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
import numpy
import os
import sys
import CGNS.DAT.db.parseCGNS as px
import CGNS.DAT.utils as ut
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
     '.MetaData/CheckSum':         numpy.array(ut.checksum(f),'c'),
     '.MetaData/GlobalCheckSum':   numpy.array('000','c'),
     '.MetaData/CreationDate':     numpy.array(dt,'c'),
     '.MetaData/ModificationDate': numpy.array(dt,'c'),
     '.MetaData/Platform':         numpy.array(os.uname()[1],'c'),
     '.MetaData/Owner':            numpy.array('BS','c'),
     '.MetaData/Policy':           numpy.array('PUBLIC','c'),
     '.MetaData/Version':          numpy.array(0,'i'),
     '.MetaData/Release':          numpy.array(0,'i'),
     '.MetaData/Change':           numpy.array(1,'i'),
     '.MetaData/Title':            numpy.array('No title set','c'),
     '.CHANCE/CaseNumber':         numpy.array('XX-0','c'),
     '.CHANCE/Family':             numpy.array('???','c'),
     '.CHANCE/Measurements':       numpy.array('???','c'),
    },
    }   
    addMeta(f,dct[id])
    addMeta(f,{})    
  except IndexError:
    print "usage: addMeta.py <base-id> <file-path>"
