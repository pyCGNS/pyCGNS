#  -------------------------------------------------------------------------
#  pyCGNS.APP - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import string

def getAllNodesByTypeList(typelist,tree):
  if (tree[3] != typelist[0]): return None
  n=getAllNodesFromTypeList(typelist[1:],tree[2],"%s"%tree[0],[])
  if (n==-1): return None
  return n

def getAllNodesFromTypeList(typelist,node,path,result):
  for c in node:
    if (c[3] == typelist[0]):
      if (len(typelist) == 1):
        result.append("%s/%s"%(path,c[0]))
      else:
        getAllNodesFromTypeList(typelist[1:],c[2],"%s/%s"%(path,c[0]),result)
  return result

def getPaths(tree,path,plist):
  for c in tree[2]:
    plist.append(path+'/'+c[0])
    getPaths(c,path+'/'+c[0],plist)
    
def getAllPaths(tree):
  plist=[]
  path=''
  getPaths(tree,path,plist)
  return plist
