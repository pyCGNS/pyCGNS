#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
"""
 We are changing for a full h5py implementation, default switch is in
 config: HAS_H5PY forces h5py instead of raw HDF5 C API implementation

 CGNS.MAP module is the interface to CGNS/HDF5 (formerly was CHLone)
  - now all the CHLone code is embedded in CGNS.MAP (EmbeddedCHLone)
  - you DO NOT need to install CHLone on its own

  Flags: 

  - python does NOT have bitfiled or unseigned int that would act as
    bit masks we have in C. So each operation on such bitfields (flags)
    should be re-masked against the actual 32bits size of the flags,
    which means you should always have a & flags.ALL in your operation
"""
#
from CGNS import backend_h5py as HAS_H5PY


if HAS_H5PY:
    from .cgio import load, save, probe, flags

    NONE = flags.NONE
    ALL = flags.ALL
    TRACE = flags.TRACE
    FOLLOWLINKS = flags.FOLLOWLINKS
    DEBUG = flags.DEBUG
    NODATA = flags.NODATA
    MERGELINKS = 0
    COMPRESS = flags.COMPRESS
    REVERSEDIMS = flags.REVERSEDIMS
    OWNDATA = flags.OWNDATA
    UPDATE = flags.UPDATE
    DELETEMISSING = flags.DELETEMISSING
    ALTERNATESIDS = flags.ALTERNATESIDS
    NOTRANSPOSE = 0
    UPDATEONLY = flags.UPDATEONLY
    FORTRANFLAG = flags.FORTRANFLAG
    PROPAGATE = flags.PROPAGATE
    DEFAULT = flags.DEFAULT
    DEFAULTS = flags.DEFAULTS
    KEEPLIST = flags.KEEPLIST
    CHECKSUM = flags.CHECKSUM

    LKOK = flags.links.OK
    LKFAIL = flags.links.FAIL
    LKBADSYNTAX = flags.links.BADSYNTAX
    LKNOFILE = flags.links.NOFILE
    LKFILENOREAD = flags.links.FILENOREAD
    LKNONODE = flags.links.NONODE
    LKLOOP = flags.links.LOOP
    LKIGNORED = flags.links.IGNORED

else:
    from .EmbeddedCHLone import load, save, probe
    from .EmbeddedCHLone import CHLoneException as error

    #
    from . import EmbeddedCHLone as CHL

    #
    S2P_NONE = CHL.FNONE
    S2P_ALL = CHL.FALL
    S2P_TRACE = CHL.FTRACE
    S2P_FOLLOWLINKS = CHL.FFOLLOWLINKS
    S2P_DEBUG = CHL.FDEBUG
    S2P_NODATA = CHL.FNODATA
    S2P_MERGELINKS = 0
    S2P_COMPRESS = CHL.FCOMPRESS
    S2P_REVERSEDIMS = CHL.FREVERSEDIMS
    S2P_OWNDATA = CHL.FOWNDATA
    S2P_UPDATE = CHL.FUPDATE
    S2P_DELETEMISSING = CHL.FDELETEMISSING
    S2P_ALTERNATESIDS = CHL.FALTERNATESIDS
    S2P_NOTRANSPOSE = 0
    S2P_UPDATEONLY = CHL.FUPDATEONLY
    S2P_FORTRANFLAG = CHL.FFORTRANFLAG
    S2P_PROPAGATE = CHL.FPROPAGATE
    S2P_DEFAULT = CHL.FDEFAULT
    S2P_DEFAULTS = CHL.FDEFAULTS
    S2P_KEEPLIST = CHL.FKEEPLIST
    S2P_CHECKSUM = CHL.FCHECKSUM

    S2P_LKOK = CHL.LKOK
    S2P_LKFAIL = CHL.LKFAIL
    S2P_LKBADSYNTAX = CHL.LKBADSYNTAX
    S2P_LKNOFILE = CHL.LKNOFILE
    S2P_LKFILENOREAD = CHL.LKFILENOREAD
    S2P_LKNONODE = CHL.LKNONODE
    S2P_LKLOOP = CHL.LKLOOP
    S2P_LKIGNORED = CHL.LKIGNORED

    NONE = CHL.FNONE
    ALL = CHL.FALL
    TRACE = CHL.FTRACE
    FOLLOWLINKS = CHL.FFOLLOWLINKS
    DEBUG = CHL.FDEBUG
    NODATA = CHL.FNODATA
    MERGELINKS = 0
    COMPRESS = CHL.FCOMPRESS
    REVERSEDIMS = CHL.FREVERSEDIMS
    OWNDATA = CHL.FOWNDATA
    UPDATE = CHL.FUPDATE
    DELETEMISSING = CHL.FDELETEMISSING
    ALTERNATESIDS = CHL.FALTERNATESIDS
    NOTRANSPOSE = 0
    UPDATEONLY = CHL.FUPDATEONLY
    FORTRANFLAG = CHL.FFORTRANFLAG
    PROPAGATE = CHL.FPROPAGATE
    DEFAULT = CHL.FDEFAULT
    DEFAULTS = CHL.FDEFAULTS
    KEEPLIST = CHL.FKEEPLIST
    CHECKSUM = CHL.FCHECKSUM

    LKOK = CHL.LKOK
    LKFAIL = CHL.LKFAIL
    LKBADSYNTAX = CHL.LKBADSYNTAX
    LKNOFILE = CHL.LKNOFILE
    LKFILENOREAD = CHL.LKFILENOREAD
    LKNONODE = CHL.LKNONODE
    LKLOOP = CHL.LKLOOP
    LKIGNORED = CHL.LKIGNORED

    # --- forward compat stuff here
    class Flags(object):
        pass

    flags = Flags()
    flags.links = Flags()

    flags.NONE = NONE
    flags.ALL = ALL
    flags.TRACE = TRACE
    flags.FOLLOWLINKS = FOLLOWLINKS
    flags.DEBUG = DEBUG
    flags.NODATA = NODATA
    flags.MERGELINKS = 0
    flags.COMPRESS = COMPRESS
    flags.REVERSEDIMS = REVERSEDIMS
    flags.OWNDATA = OWNDATA
    flags.UPDATE = UPDATE
    flags.DELETEMISSING = DELETEMISSING
    flags.ALTERNATESIDS = ALTERNATESIDS
    flags.NOTRANSPOSE = 0
    flags.UPDATEONLY = UPDATEONLY
    flags.FORTRANFLAG = FORTRANFLAG
    flags.PROPAGATE = PROPAGATE
    flags.DEFAULT = DEFAULT
    flags.DEFAULTS = DEFAULTS
    flags.KEEPLIST = KEEPLIST
    flags.CHECKSUM = CHECKSUM

    flags.links.OK = LKOK
    flags.links.FAIL = LKFAIL
    flags.links.BADSYNTAX = LKBADSYNTAX
    flags.links.NOFILE = LKNOFILE
    flags.links.FILENOREAD = LKFILENOREAD
    flags.links.NONODE = LKNONODE
    flags.links.LOOP = LKLOOP
    flags.links.IGNORED = LKIGNORED


def flags_set(source=flags.DEFAULT, flag=flags.NONE):
    return source | flag & flags.ALL


def flags_unset(source, flag):
    return source & ~flag & flags.ALL


def flags_check(source, flag):
    return source & flag == flag


# --- last line
