
import string

def checkPath(path):
  return 1

def getValueByPath(path,tree): # not bulletproof
  n=getNodeFromPath(path.split('/'),tree)
  if (n==-1): return None
  return n[1]

def getNodeByPath(path,tree):
  if (not checkPath(path)): return None
  if (path[0]=='/'): path=path[1:]
  n=getNodeFromPath(path.split('/'),tree)
  if (n==-1): return None
  return n

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

def getNodeFromPath(path,node):
  for c in node[2]:
    if (c[0] == path[0]):
      if (len(path) == 1): return c
      return getNodeFromPath(path[1:],c)
  return -1
