#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
import CGNS
import CGNS.PAT.cgnskeywords as CK
import CGNS.PAT.cgnstypes as CT
import CGNS.PAT.cgnserrors as CE
import CGNS.PAT.cgnsutils as CU

import numpy as NPY


# -----------------------------------------------------------------------------
class CGNSPythonChildren(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            for c in self:
                if c.name == key:
                    return c
        else:
            return list.__getitem__(self, key)


# -----------------------------------------------------------------------------
class CGNSPython:
    """
    A CGNS/Python object
    """

    def __init__(self, node, parent=None):
        self.__node = node
        self.__parent = parent

    @property
    def name(self):
        """
        Name of the node (node[0])
        """
        return self.__node[0]

    @property
    def sidstype(self):
        return self.__node[3]

    @property
    def sids(self):
        return self.sidstype

    @property
    def value(self):
        return self.data

    @property
    def data(self):
        return self.__node[1]

    @property
    def child(self):
        l = CGNSPythonChildren([CGNSPython(n) for n in self.__node[2]])
        for n in l:
            n.parent = self
        return l

    @property
    def children(self):
        return self.__node[2]

    def nextChild(self, sidstype=None, namepattern=None):
        for c in self.__node[2]:
            take = False
            if sidstype is not None:
                if (isinstance(sidstype, list) and (c[3] in sidstype)) or (
                    isinstance(sidstype, str) and (c[3] == sidstype)
                ):
                    take = True
            else:
                take = True
            if take:
                n = CGNSPython(c)
                n.parent = self
                yield n

    @property
    def node(self):
        return self.__node

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, node):
        if isinstance(node, CGNSPython):
            self.__parent = node
        else:
            self.__parent = CGNSPython(node)

    def __str__(self):
        return CU.toString(self.__node)

    def __len__(self):
        return len(self.__node[2])


# ---
