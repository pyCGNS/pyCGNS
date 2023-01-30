#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
from .. import cgnslib as C
from .. import cgnserrors as E
from .. import cgnskeywords as K
import numpy as N

data = C.newAdditionalFamilyName(None, family="{AdditionalFamilyName}")
status = "-"
comment = "SIDS Leaf node"
pattern = [data, status, comment]
