# -*- coding: utf-8 -*-
#
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from ..APP.lib.queries import asQuery
from ..APP.lib.queries import runQuery


@asQuery
def nodeNameSearch(C):
    """
    Search by
    Node name

    Search all nodes with the exact NAME as argument.

    The argument name need not to be a tuple or to have quotes,
    all the following values are ok and would match the NAME <i>ZoneType</i>:

    ZoneType
    'ZoneType'
    ('ZoneType',)
    """
    if C.NAME == C.ARGS[0]:
        return C.PATH


@asQuery
def nodeTokenSearch(C):
    """
    Search by
    Node token

    Search all nodes with NAME as a token in the PATH

    The argument name need not to be a tuple or to have quotes,
    all the following values are ok and would match the NAME <i>ZoneBC</i>:

    ZoneBC
    'ZoneBC'
    ('ZoneBC',)
    """
    if C.ARGS[0] in C.PATH:
        return C.PATH


FILE = "HYB/vbv-part32_comp_period_links_dom_32_SUB.hdf"
ARGS = ["GridLocation"]

# -----------------------------------------------------------------
from .. import MAP as CGM

(t, l, p) = CGM.load(FILE)
print(runQuery(t, l, p, nodeNameSearch, ["GridLocation"]))
print(runQuery(t, l, p, nodeTokenSearch, ["Zone"]))
