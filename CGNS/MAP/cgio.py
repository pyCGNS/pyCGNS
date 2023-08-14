#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import numpy
import h5py
import os

from ..PAT import cgnskeywords as CGK
from ..PAT import cgnsutils as CGU
from ..PAT import cgnserrors as CGE


class Flags(object):
    pass


flags = Flags()
flags.NONE = 0x00000000
flags.ALL = 0xFFFFFFFF
flags.TRACE = 0x00000001
flags.FOLLOWLINKS = 0x00000002
flags.NODATA = 0x00000004
flags.KEEPLIST = 0x00000008
flags.MERGELINKS = 0
flags.COMPRESS = 0x00000010
flags.REVERSEDIMS = 0x00000020
flags.OWNDATA = 0x00000040
flags.UPDATE = 0x00000080
flags.DEBUG = 0x00008000
flags.DELETEMISSING = 0x00000100
flags.ALTERNATESIDS = 0x00000200
flags.NOTRANSPOSE = 0
flags.UPDATEONLY = 0x00000400
flags.FORTRANFLAG = 0x00000800
flags.PROPAGATE = 0x00004000
flags.LINKOVERRIDE = 0x00010000
flags.DEFAULT = (
    flags.FOLLOWLINKS
    | flags.DELETEMISSING
    | flags.OWNDATA
    | flags.REVERSEDIMS
    | flags.FORTRANFLAG
    | flags.OWNDATA
    | flags.LINKOVERRIDE
)
flags.DEFAULTS = flags.DEFAULT
flags.CHECKSUM = 0x00080000

flags.links = Flags()
flags.links.OK = 0x0000
flags.links.FAIL = 0x0001
flags.links.BADSYNTAX = 0x0002
flags.links.NOFILE = 0x0004
flags.links.FILENOREAD = 0x0008
flags.links.NONODE = 0x0010
flags.links.LOOP = 0x0020
flags.links.IGNORED = 0x0040
flags.links.UPDATED = 0x0080


def raiseException(code, *args):
    if args:
        raise MAPException((code, MAPException.mTable[code].format(args)))
    else:
        raise MAPException((code, MAPException.mTable[code]))


def probe(filename, path=None):
    tfile = os.path.normpath(os.path.expanduser(filename))
    if not os.path.isfile(tfile):
        raiseException(900, tfile)
    return True


def load(filename, nodata=False, maxdata=65, subtree="", follow_links=False, **kwargs):
    """
    Load a CGNS/HDF5 CGNS tree as a CGNS/Python object::

      from CGNS.MAP import load

      (tree, links, paths) = load('cgnsfile.cgns')

    Detailed documentation about the load can be found
    `here <https://pycgns.github.io/MAP/_index.html>`_, the following doc only
    describes the function signature.

    :arg str filename: CGNS/HDF5 file to open
    :arg bool nodata: if `True` the actual node data is load only if below the `maxdata`
    :arg int maxdata: if `nodata` is `True`, `maxdata` gives the threshold for loading actual data
    :arg str subtree: loads only starting from the `subtree` path arg
    :arg bool follow_links: If `True` follows the linked-to files (default:`False`)

    :return: a tuple (tree, links, paths) see remaks below

    :Remarks:
      - `tree` is a Python list, the actual CGNS/Python tree created from the CGNS/HDF5 file.
      - `links` is the list of linked-to files and nodes used during the `tree` creation.
      - `paths` is the list of nodes paths in `tree` you have to take care of such as no-data nodes.

    """

    def _load(grp, links, paths, load_children=True):
        if grp.attrs["type"] == "LK":
            if not follow_links:
                links.append(
                    [
                        "",
                        grp[" file"].value[:-1].tostring().decode("ascii"),
                        grp[" path"].value[:-1].tostring().decode("ascii"),
                        grp.name,
                        64,
                    ]
                )
                return None
            else:
                grp = grp[" link"]

        node = [str(grp.attrs["name"]), None, [], str(grp.attrs["label"])]

        if " data" in grp:
            dset = grp[" data"]
            n = dset.size
            if not nodata or n < maxdata:
                load_data = True
            else:
                load_data = False
                paths.append([grp.name, 1, grp.attrs["type"], dset.shape])

            if load_data:
                node[1] = numpy.copy(dset[()].transpose(), order="F")
                if grp.attrs["type"] == b"C1":
                    node[1] = numpy.vectorize(chr)(node[1]).astype(numpy.dtype("c"))

        if load_children:
            cnames = [x for x in grp.keys() if x[0] != " "]
            for cname in cnames:
                child = _load(grp[cname], links, paths)
                if child is not None:
                    node[2].append(child)

        return node

    links = []
    paths = []
    with h5py.File(filename, "r") as h5f:
        if subtree == "":
            tree = _load(h5f, links, paths)
        else:
            grp = h5f[subtree]
            node = _load(grp, links, paths)
            while grp.name != "/":
                pnode = _load(grp.parent, links, paths, load_children=False)
                pnode[2].append(node)
                node = pnode
                grp = grp.parent

            tree = node

    tree[0] = "CGNSTree"
    tree[3] = "CGNSTree_t"

    return tree, links, paths


def save(filename, tree, links=[], **kwargs):
    def _cst_size_str(s, n):
        return numpy.bytes_(bytes(s, "ascii").ljust(n, b"\x00"))

    def _save(grp, node):
        _cst_size_str(node[0], 33)

        if node[3] == "CGNSTree_t":
            grp.attrs["name"] = _cst_size_str("HDF5 MotherNode", 33)
            grp.attrs["label"] = _cst_size_str("Root Node of HDF5 File", 33)
        else:
            grp.attrs["name"] = _cst_size_str(node[0], 33)
            grp.attrs["label"] = _cst_size_str(node[3], 33)
            grp.attrs["flags"] = numpy.array([0], dtype=numpy.int32)

        if node[1] is None:
            grp.attrs["type"] = _cst_size_str("MT", 3)
        else:
            if node[1].dtype == numpy.float32:
                grp.attrs["type"] = _cst_size_str("R4", 3)
            elif node[1].dtype == numpy.float64:
                grp.attrs["type"] = _cst_size_str("R8", 3)
            elif node[1].dtype == numpy.int32:
                grp.attrs["type"] = _cst_size_str("I4", 3)
            elif node[1].dtype == numpy.int64:
                grp.attrs["type"] = _cst_size_str("I8", 3)
            elif node[1].dtype == numpy.dtype("|S1"):
                grp.attrs["type"] = _cst_size_str("C1", 3)
            else:
                raise NotImplementedError("unknown dtype %s" % node[1].dtype)

            data = node[1]
            if node[1].dtype == numpy.dtype("|S1"):
                idx1 = numpy.nonzero(data != b"")
                idx2 = numpy.nonzero(data == b"")
                tmp = data.copy()
                data = numpy.zeros(data.shape, dtype=numpy.int8)
                data[idx1] = numpy.vectorize(ord)(tmp[idx1])
                data[idx2] = 0
                data = data.astype(numpy.int8)

            data_view = data.T
            grp.create_dataset(" data", data=data_view)

        for child in node[2]:
            sgrp = grp.create_group(child[0], track_order=True)
            _save(sgrp, child)

    def _add_links(h5f, links):
        for dname, filename, rpth, lpth, n in links:
            tmp = lpth.split("/")
            name = tmp[-1]
            ppth = "/".join(tmp[:-1])

            grp = h5f[ppth]
            sgrp = grp.create_group(name)

            sgrp.attrs["name"] = _cst_size_str(name, 33)
            sgrp.attrs["label"] = _cst_size_str("", 33)
            sgrp.attrs["type"] = _cst_size_str("LK", 3)

            data = numpy.array([ord(x) for x in rpth + "\x00"], dtype=numpy.int8)
            sgrp.create_dataset(" path", data=data)
            data = numpy.array([ord(x) for x in filename + "\x00"], dtype=numpy.int8)
            sgrp.create_dataset(" file", data=data)
            sgrp[" link"] = h5py.ExternalLink(filename, rpth)

            assert dname == ""

    with h5py.File(filename, "w", libver=("v108", "v108"), track_order=True) as h5f:
        h5f.create_dataset(
            " format", data=numpy.array([78, 65, 84, 73, 86, 69, 0], dtype=numpy.int8)
        )
        h5f.create_dataset(
            " hdf5version",
            data=numpy.array(
                [
                    72,
                    68,
                    70,
                    53,
                    32,
                    86,
                    101,
                    114,
                    115,
                    105,
                    111,
                    110,
                    32,
                    49,
                    46,
                    56,
                    46,
                    49,
                    53,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=numpy.int8,
            ),
        )
        _save(h5f, tree)
        _add_links(h5f, links)


#
# --- last line
