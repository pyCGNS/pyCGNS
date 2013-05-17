#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS
import CGNS.PAT.cgnskeywords as CK
import CGNS.PAT.cgnstypes    as CT
import CGNS.PAT.cgnserrors   as CE
import CGNS.PAT.cgnsutils    as CU

import numpy as NPY

# -----------------------------------------------------------------------------
class CGNSPythonChildren(list):
  def __getitem__(self,key):
      if (type(key) is str):
          for c in self:
              if (c.name==key): return c
      else:
          return list.__getitem__(self,key)

# -----------------------------------------------------------------------------
class CGNSPython(object):
  def __init__(self,node,parent=None):
    self.__node=node
    self.__parent=parent
  @property
  def name(self):
    return self.__node[0]
  @property
  def type(self):
    return self.__node[3]
  @property
  def data(self):
    return self.__node[1]
  @property
  def child(self):
    l=CGNSPythonChildren([CGNSPython(n) for n in self.__node[2]])
    for n in l:
       n.parent=self
    return l
  @property
  def children(self):
    return self.__node[2]
  def nextChild(self):
    for c in self.__node[2]:
       n=CGNSPython(c)
       n.parent=self
       yield n
  @property 
  def parent(self):
    return self.__parent
  @parent.setter
  def parent(self,node):
    if (type(node).__name__=='CGNSPython'):
      self.__parent=node
    else:
      self.__parent=CGNSPython(node)
  def __str__(self):
    return CU.toString(self.__node)
  def __len__(self):
    return len(self.__node[2])

# ---
