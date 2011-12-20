#  -------------------------------------------------------------------------
#  pyCGNS.MAP - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
# now all the code is in CHLone and its CGNS/Python interface
#
from CHLone import load
from CHLone import save
from CHLone import CHLoneException as error
#
import CHLone

S2P_NONE=CHLone.FNONE
S2P_ALL=CHLone.FALL
S2P_TRACE=CHLone.FTRACE
S2P_FOLLOWLINKS=CHLone.FFOLLOWLINKS
S2P_NODATA=CHLone.FNODATA
S2P_MERGELINKS=CHLone.FMERGELINKS
S2P_COMPRESS=CHLone.FCOMPRESS
S2P_REVERSEDIMS=CHLone.FREVERSEDIMS
S2P_OWNDATA=CHLone.FOWNDATA
S2P_UPDATE=CHLone.FUPDATE
S2P_DELETEMISSING=CHLone.FDELETEMISSING
S2P_ALTERNATESIDS=CHLone.FALTERNATESIDS
S2P_NOTRANSPOSE=CHLone.FNOTRANSPOSE
S2P_FORTRANFLAG=CHLone.FFORTRANFLAG
S2P_DEFAULT=CHLone.FDEFAULT
S2P_DEFAULTS=CHLone.FDEFAULTS
# --- last line
