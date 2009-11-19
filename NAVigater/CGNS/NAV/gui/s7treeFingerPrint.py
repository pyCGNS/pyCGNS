#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $File$
#  $Node$
#  $Last$
#  -------------------------------------------------------------------------
import s7globals
G___=s7globals.s7G

import CGNS.NAV.supervisor.s7error   as E

# ----------------------------------------------------------------------
def encapsulateTreeOrSubTree(tree):
  #
  # check different flavours of tree roots
  treetype=0
  stree=None

  # TESTS ORDER IS SIGNIFICANT

  if (tree == None): return None

  # -- A standard Python/CGNS tree
  #    pattern is ['CGNSTree',None,[<children-list>],'CGNSTree_t']
  #    -> take [<children-list>]
  if ( not treetype
       and (len(tree) == 4)
       and (type(tree) == type([]))
       and (tree[0] == 'CGNSTree')
       and (tree[3] == 'CGNSTree_t')): treetype=1

  # -- A Python/CGNS tree starting with bases nodes and CGNSLibraryVersion
  #    pattern is [[<CGNSLibraryVersion>],[<base>],[<base>,...]
  #    -> take tree as it is
  if ( not treetype
       and (type(tree) == type([])) ):

    for nt in tree:
      if (     (nt != None)
           and (len(nt) == 4)
           and (nt[3] == 'CGNSLibraryVersion_t')): treetype=4

  # -- A Python/CGNS tree starting from base node, no CGNSLibraryVersion
  #    pattern is [<base-name>,<base-dims>,[<children-list>],'CGNSBase_t']
  #    -> put into list to make [<children-list>] pattern
  if ( not treetype
       and (len(tree) == 4)
       and (type(tree) == type([]))
       and (tree[3] == 'CGNSBase_t')): treetype=3
  
  # -- A sub-tree which gives its children list to parse
  #    the pattern is [None, None, [<children-list>], None]
  #    -> take [<children-list>]
  if ( not treetype
       and (len(tree) == 4)
       and (type(tree) == type([]))
       and (tree[0] == None)
       and (type(tree[2]) == type([]))): treetype=2

  # -- A pattern which gives its node template
  #    the pattern is [<name>, <data>, [<children-list>], <type>]
  #    -> put into list to make [<children-list>] pattern
  if ( not treetype
       and (len(tree) == 4)
       and (type(tree) == type([]))
       and (type(tree[0]) == type(""))
       and (type(tree[3]) == type(""))
       and (tree[3][-2:]  == "_t")
       and (type(tree[2]) == type([]))): treetype=5

  # -- Another pattern whith more than one child
  #    the pattern is [ [<name>, <data>, [<children-list>], <type>], ...]
  #    -> put into list to make [<children-list>] pattern
  if ( not treetype
       and (type(tree) == type([]))
       and (len(tree) > 0)
       and (type(tree[0][0]) == type(""))
       and (type(tree[0][3]) == type(""))
       and (tree[0][3][-2:]  == "_t")
       and (type(tree[0][2]) == type([]))): treetype=6

  if (treetype == 1):  stree=tree[2]
  if (treetype == 2):  stree=tree[2]
  if (treetype == 3):  stree=[tree]
  if (treetype == 4):  stree=tree
  if (treetype == 5):  stree=[tree]
  if (treetype == 6):  stree=tree

  return stree

# ----------------------------------------------------------------------
def maxDepthAux(node,depth,globalmax):
  if (node and (len(node)>1) and node[2]):
    depth+=1
    for n in node[2]:
      globalmax=max(maxDepthAux(n,depth,globalmax),depth)
  return globalmax

# ----------------------------------------------------------------------
# One instance per view, actual CGNS tree is in wTreeFingerPrint
class wTreeView:
  def __init__(self):
    self.view=None
    self.node='/'
    self.id=0
    self.type='T'
    
# ----------------------------------------------------------------------
# One instance per tree, for instance a file
# Many views can be opened on the same tree, viewlist is the ordered list
# of views for the tree
class wTreeFingerPrint:
  def __init__(self,filedir,filename,tree,node=None):
    self.status=1
    if (filedir): self.filedir=filedir
    else:         self.filedir='.'
    self.filename=filename
    stree=encapsulateTreeOrSubTree(tree)
    if (stree==None): self.status=0
    self.nviews=0
    self.viewlist=[]
    self.tree=[None, None, stree, None] # fake root node
    self.links=[]
    self.state=' '
    self.comment=''
    self.keyword=''
    self.profdir=None
    self.fileext='.py'
    self.nosave=0
  def saveFileName(self):
    return "%s/%s%s"%(self.filedir,self.filename,self.fileext)
  def isModified(self):
    if (self.state==' '): return 0
    return 1
  def unmodified(self):
    self.state=''
  def modified(self):
    self.state='M'
  def addView(self,view,node,vtype):
    newview=wTreeView()
    self.nviews+=1
    newview.id=self.nviews
    if (node): newview.node=node
    else:      newview.node='/'
    newview.type=vtype
    newview.view=view
    self.viewlist.append(newview)
    return newview.id

  def maxDepth(self):
    return maxDepthAux(self.tree,0,0)
    
# ----------------------------------------------------------------------
