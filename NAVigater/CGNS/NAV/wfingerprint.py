#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
from PySide.QtCore       import *
from PySide.QtGui        import *
import os.path
import CGNS.MAP

import CGNS.NAV.wmessages as MSG

class Q7fingerPrint:
    __viewscounter=0
    __extension=[]
    @classmethod
    def treeLoad(cls,control,selectedfile):
        f=selectedfile
        (filedir,filename)=(os.path.normpath(os.path.dirname(f)),
                            os.path.basename(f))
        try:
            (tree,links)=CGNS.MAP.load(f,CGNS.MAP.S2P_DEFAULT)
        except CGNS.MAP.error:
            MSG.message("Cannot open file:",filedir+'/'+filename,MSG.WARNING)
            return None
        return Q7fingerPrint(control,filedir,filename,tree,links)
    @classmethod
    def closeAllTrees(cls):
        for x in cls.__extension: x.closeAllViews()
    def __init__(self,control,filedir,filename,tree,links,**kw):
        self.filename=filename
        self.tree=tree
        self.filedir=filedir
        self.links=links
        self.model=None
        self.depth=0
        self.views={}
        self.control=control
        Q7fingerPrint.__extension.append(self)
    def addChild(self,viewtype,view):
        Q7fingerPrint.__viewscounter+=1
        if not self.views.has_key(viewtype): self.views[viewtype]=[]
        self.views[viewtype].append(view)
        return Q7fingerPrint.__viewscounter
    def closeAllViews(self):
        for vtype in self.views:
            for v in self.views[vtype]: v.close()
            
    
