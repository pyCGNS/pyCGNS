#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
# Change these values to fit your installation
#
# --- If you leave empty the variables below the build process would
#     try to guess for you, using your current environment variables.
#     Change these values only if you need to add/change something.
#
# --- stuff to add for numpy (used by NAV)
#     you should not change this, the numpy stuff is expected to be detected
#     using the numpy import itself
#

if False:
    NUMPY_VERSION = ""
    NUMPY_PATH_INCLUDES = []
    NUMPY_PATH_LIBRARIES = [""]
    NUMPY_LINK_LIBRARIES = [""]
    NUMPY_EXTRA_ARGS = []
#
# --- stuff to add for HDF5 (used by MAP)
#     the hdf5 install is detected using a 'which h5dump' and then we parse the
#     installation. Thus non-standard installs would require to set these vars
#
if True:
    HDF5_VERSION = ""
    HDF5_PATH_INCLUDES = [
        "",
    ]
    HDF5_PATH_LIBRARIES = [
        "",
    ]
    HDF5_LINK_LIBRARIES = ["hdf5"]
    HDF5_EXTRA_ARGS = ["-Wno-return-type"]

# -------------------------------------------------------------------------
