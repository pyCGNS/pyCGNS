# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 45 $ $Date: 2007-12-20 15:26:49 +0100 (Thu, 20 Dec 2007) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source 
# tree for license information. 

import CGNS.hdf
import numpy as N
import cgnskeywords as K
import cgnslib      as L
import os.path 

CGNSHDF_CGNSLibraryVersion=[-1,K.CGNSLibraryVersion_s,
                               K.CGNSLibraryVersion_ts,
                               "R4",N.array([2.4],dtype=N.float32)]

# --------------------------------------------------
def getNodeType(node):
  data=node[1]
  if (node[0] == 'CGNSLibraryVersion_t'):
    return 'R4' # ONLY ONE R4 IN ALL SIDS !
  if ( data == None ):
    return 'MT'
  if (type(data) == type("")):
    return 'C1'
  if ( (type(data) == type(1.2))):
    return 'R8'
  if ( (type(data) == type(1))):
    return 'I4'
  if ( (type(data) == type(N.ones((1,)))) ):
    if (data.dtype.char in ['S','c']):        return 'C1'
    if (data.dtype.char in ['f','F']):        return 'R4'
    if (data.dtype.char in ['D','d']):        return 'R8'
    if (data.dtype.char in ['l','i','I']):    return 'I4'
  if ((type(data) == type([])) and (len(data))): # oups !
    if (type(data[0]) == type("")):           return 'C1' 
    if (type(data[0]) == type(0)):            return 'I4' 
    if (type(data[0]) == type(0.0)):          return 'R8'
  return '??'

# --------------------------------------------------
def asNumType(node):
  data=node[1]
  if (node[0]=='CGNSLibraryVersion_t'): return N.array(data,dtype=N.float32)
  if (data==None):                      return None
  if ((type(data)==type("")) and (data=="")):
                                        return N.array([' '],dtype='c')
  if (type(data)==type("")):            return N.array(data,dtype='c')
  if ((type(data)==type(1.2))):         return N.array([data],dtype=N.float64)
  if ((type(data)==type(1))):           return N.array([data],dtype=N.int32)

  if ( (type(data) == type(N.ones((1,)))) ):
    if (data.dtype.char in ['S']):  return N.array(data,dtype='c')

  if ((type(data) == type([])) and (len(data))): # oups !
      pass

  return data

def checkPath(path):
  return 1

# -----------------------------------------------------------------------------
class CGNSHDF5tree:
  # ------------------------------------------------------
  def __init__(self,filename,mode,flags=0,noversion=None):
    self.flags=flags|CGNS.hdf.WITHDATA
    self.db=None
    if (filename):
      self.filename=os.path.basename(filename)
      self.dirname=os.path.dirname(filename)
      self.db=CGNS.hdf.open(filename,mode,flags)
      if ((mode & CGNS.hdf.NEW) and (not noversion)):
          self.db.create(CGNSHDF_CGNSLibraryVersion,self.db.root,self.flags)
  # ------------------------------------------------------
  def asHDFNode(self,node):
    return [-1, node[0], node[3], getNodeType(node), asNumType(node)]
  # ------------------------------------------------------
  def addNode(self,node,parent=None):
    if (self.db == None): return None
    if (parent == None): parent=self.db.root
    return self.db.create(self.asHDFNode(node),parent,self.flags)
  # ------------------------------------------------------
  def addTree(self,node,parent=None):
    if (self.db == None): return None
    if (parent == None): parent=self.db.root
    id=self.db.create(self.asHDFNode(node),parent,self.flags)
    for cnode in node[2]:
      self.addTree(cnode,id)
    return id
  # ------------------------------------------------------
  def close(self):
    if (self.db == None): return None
    self.db.close()
    self.db=None
  # ------------------------------------------------------
  # [ id, name, label, datatype, data, children ]
  def retrieve(self,id=None):
    flg=self.flags|CGNS.hdf.WITHCHILDREN
    if (id == None): r=self.db.retrieve(self.db.root,flg)
    else:            r=self.db.retrieve(id,flg)
    return r
  # ------------------------------------------------------
  # input:  parent, path
  # output: [ id, name, label, datatype, data, children ]
  def find(self,parent,path):
    if (self.db == None): return None
    if (parent == None):  parent=self.db.file
    flg=self.flags|CGNS.hdf.WITHCHILDREN
    nodeid=None
    if (checkPath(path)):
      nodeid=self.db.find(parent,path,flg)
    if (nodeid == None): return None
    node=self.retrieve(nodeid)
    return node[0]
  # ------------------------------------------------------
  # link update is a delete/new, because there is no recursion
  def link(self,parentid,name,destfile,destname):
    if (self.db == None): return None
    if (parentid==None): parentid=self.db.root
    flg=self.flags|CGNS.hdf.TRACE
    return self.db.link(parentid,name,destfile,destname,flg)
  # ------------------------------------------------------
  def update(self,node):
    if (self.db == None): return None
    flg=self.flags|CGNS.hdf.TRACE
    return self.db.update(node,flg)    
  # ------------------------------------------------------
  def move(self,parentid,name,destid,newname):
    if (self.db == None): return None
    flg=self.flags|CGNS.hdf.TRACE
    return self.db.move(parentid,name,destid,newname,flg)    
  # ------------------------------------------------------
  def delete(self,parentid,name):
    if (self.db == None): return None
    flg=self.flags|CGNS.hdf.TRACE
    return self.db.delete(parentid,name,flg)    
  # ------------------------------------------------------
  def save(self,tree):
    rt=tree
    if (tree[0] == 'HDF5 MotherNode'):
      for n in tree[2]:
        if (n[3] == K.CGNSBase_ts): self.addTree(n)
    return rt
  # ------------------------------------------------------
  def getChildren(self,node):
    r=[]
    for c in node[-1]:
      r.append(c[0])
    return r
  # ------------------------------------------------------
  def loadOne(self,parent,lid,flags):
    for inode in lid:
      lnode=self.db.retrieve(inode,flags)
      pnode=L.newNode(lnode[1],lnode[4],[],lnode[2],parent)
      self.loadOne(pnode,self.getChildren(lnode),flags)
  # ------------------------------------------------------
  def load(self,pathorid=None):
    flg=self.flags
    flg|=CGNS.hdf.WITHCHILDREN    
    flg|=CGNS.hdf.WITHDATA
    t=None
    if (pathorid == None): top=self.db.root
    else:
      if    (type(pathorid) == type(1)): top=pathorid
      elif ((type(pathorid) == type("")) and (checkPath(pathorid))):
        top=self.db.find(self.db.file,pathorid,flg)
      else: raise "Bad arg type"
    lnode=self.db.retrieve(top,flg)
    print lnode
    tree=L.newNode(lnode[1],lnode[4],[],lnode[2])
    self.loadOne(tree,self.getChildren(lnode),flg)
    return tree
  # ------------------------------------------------------  

linkdoc="""
 links
 - the link concept doesn't exist in a CGNS/Python tree
 - the link exists when you read/write a CGNS/Python tree on an HDF5 file
 - a link is represented by a tuple:
        ('filename','target-node-in-the-file')
 - reading an HDF5 file:
       -in all cases, the children list returned by 'retrieve' (with the
        flag WITHCHILDREN set) contains the link information
        [ [list-of-actual-node-id], [list-of-links] ]
       -if FOLLOWLINK is set (default) then any read of an HDF5 link is
        automatically transformed in a plain node. The behavior of any
        hdfmodule function is performed as if there was only one file with
        all levels of link merged
        [ [n1, n2, n3, n4, n5, n6],[None, None, lk1, None, lk2, None] ]
        the nodes n3 and n5 are actual node ids once the link is traversed,
        the information about the links are in lk1 and lk2
       -if ~FOLLOWLINK is set, then any read of an HDF5 link is ignored, the
        linked-to node doesn't exist
        [ [n1, n2, None, n4, None, n6],[None, None, lk1, None, lk2, None] ]
 - writing an HDF5 file:
       -you have to give information about the nodes which are to be taken
        as links. Such an information is given to the 'link' function on
        an existing node (or sub-tree). The 'link' function operates on a node
        and transform some of its children as links (in the case these are not
        already links).
        Once you have your children id list for a given parent, you add
        the associated link list (same as the one given by 'retrieve' with
        the FOLLOWLINK flag set on)
        [ [n1, n2, n3, n4, n5, n6],[None, None, lk1, None, lk2, None] ]
        If an existing link is now found as a plain node, the node is
        merged to the current tree if MERGELINKS is set on.
        [ [n1, n2, n3, n4, n5, n6],[None, None, None, None, lk2, None] ]

 - end-user doesn't ant to deal with such lists and checks. Thus, the interface
   hides the system. The end-user fuction are detailled below

 functions:

   saveTree(tree)
   - saves a complete tree
   
   saveTree(tree,linklist)
   - saves the tree unless it finds a links that corresponds to one of its node
     and it creates a link at this place.

   updateTree(tree)
   - saves a complete tree, takes into account existing links

   updateTree(tree,linklist)
   - saves a complete tree, takes into account existing links but add and/or
     overwrites links if the linklist is different than existing
"""
