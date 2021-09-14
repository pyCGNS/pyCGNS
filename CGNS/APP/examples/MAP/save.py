#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS.APP - Python package for CFD General Notation System - APPlicater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.MAP
from CGNS.PAT.cgnskeywords import *
import numpy
import os

filename = "/tmp/T0.cgns"

try:
    os.unlink(filename)
except os.error:
    pass

# this is a small example, because it's quite hard to read!
tree = [
    CGNSTree_s,
    None,
    [
        [
            CGNSLibraryVersion_s,
            numpy.array([2.4], dtype=numpy.float32),
            [],
            CGNSLibraryVersion_ts,
        ],
        [
            "Helicopter",
            numpy.array([3, 3], dtype=numpy.int32),
            [
                [
                    "ReferenceState",
                    None,
                    [
                        ["AngleofAttack", numpy.array([7.0]), [], DataArray_ts],
                        ["BetaAngle", numpy.array([0.0]), [], DataArray_ts],
                        ["Coef_Area", numpy.array([0.025]), [], DataArray_ts],
                    ],
                    ReferenceState_ts,
                ],
            ],
            CGNSBase_ts,
        ],
    ],
    CGNSTree_ts,
]

CGNS.MAP.save(filename, tree, links=[], flags=CGNS.MAP.S2P_DEFAULT)

# --- last line
