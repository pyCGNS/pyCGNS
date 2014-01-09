#!/usr/bin/env python
import CGNS.PAT.cgnsutils as CGU
import CGNS.MAP as CGM

"""
  cgGrep [options] file1.hdf file2.hdf ...

  -n <node name>
  -t <node type>
  -d <node data>
  -l <link path>
  -c : leaf only, do not propagate to subtrees (cut on find)
  
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name",dest="name")
parser.add_argument('-c','--cut',action='store_true')
parser.add_argument('files',nargs=argparse.REMAINDER)

args=parser.parse_args()

class Query(object):
    def __init__(self):
        self.name=None
        self.sidstype=None
        
def openFile(filename):
    flags=CGM.S2P_NODATA
    (t,l,p)=CGM.load(filename,flags=flags,maxdata=20)
    return (t,l,p,filename)

def search(T,Q,R):
  paths=CGU.getAllPaths(T)
  for path in paths:
    skip=False
    if (Q.cut):
      for cpath in R:
        if (cpath in path):
            skip=True
    if (not skip):
      if (Q.name and (Q.name in CGU.getPathToList(path,nofirst=True))):
          R.append(path)

P=[]
for F in args.files:
  T=openFile(F)
  R=[]
  Q=Query()
  Q.name=args.name
  Q.cut=args.cut
  search(T[0],Q,R)
  for p in R:
    P.append('%s:%s'%(T[3],p))
for p in P:
  print p
    
