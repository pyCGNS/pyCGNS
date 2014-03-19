#  -------------------------------------------------------------------------
#  pyCGNS.APP - Python package for CFD General Notation System - APPlicater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

MAP demos
~~~~~~~~~

The data files required for input are into the ``data`` directory.
The temporary files are created into ``/tmp`` and you have to clean
it by yourself.

The first demo to look at are ``load.py`` and ``save.py``. The first
reads a *CGNS/HDF* file and returns the corresponding *CGNS/Python* tree.
The second has a *CGNS/Python* tree and produces a new *CGNS/HDF5* file.

The ``links.py`` demo shows how to parse links or to avoid parsing them
during the *load*. It shows how to write a file with links or without
them, for example in order to merge linked files into a single file.
The use of the *link search path* is also shown.

The ``nodepernode.py`` demo shows how to load a file skeleton, without data,
and then ask for a specific node data. This is useful when you want to
load very large files. Also in this demo you can check that actual data is
returned in existing *numpy* arrays you already have allocated.

The ``dictionnary.py`` is a simple demo about *MAP* constants.

# -------------------------------------------------------------------------
