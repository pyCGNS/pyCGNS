#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import os
import sys
import string
import argparse
import glob
import re

from setuptools import setup, Extension
from pathlib import Path

# in the case the default python/distutils compiler fails with mpi, set
# this variable. You should check this compiler is the one used for HDF5 prod
# in the share/cmake config of your HDF5 installation.
CYTHON_COMPILER_FOR_MAP = "mpicc"

# --- get overall configuration
sys.path = ["{}/lib".format(os.getcwd())] + sys.path
from setuputils import (
    MAJOR_VERSION,
    MINOR_VERSION,
    HAS_MSW,
    log,
    line,
    is_windows,
    fix_path,
    search,
    clean,
    touch,
    ConfigException,
    updateVersionInFile,
    installConfigFiles,
)


line("pyCGNS v{}.{} install".format(MAJOR_VERSION, MINOR_VERSION))
line()

doc1 = """
  pyCGNS installation setup 
  - Usual python setup options are unchanged (build, install, help...)
  - The recommanded way to build is to set your shell environment with
    your own PATH, LD_LIBRARY_PATH and PYTHONPATH so that the setup would
    find all expected ressources itself.
  - You can either use the command line args as described below or edit
    the setup_userConfig.py with the values you want.

  All packages are installed if all expected dependancies are found.
  See doc for more installation details and depend:
  https://pycgns.github.io/install.html

  MPI: using HDF5 with parallel features adds dependancies on mpi. The
  simple way to resolve these deps is to force mpicc instead of cc:

  CC=mpicc python setup.py build

  or edit the pyCGNS/stup.py file and change CYTHON_COMPILER_FOR_MAP

  ** IMPORTANT WARNING **
  The *install* command runs the *build* as first step.
  If you run first the *build* with specific options
  you *SHOULD* add these options again in the *install*
  command line unless you will have a *NEW* build.

"""

doc2 = """
  Examples:

  1. The best way is to let setup find out required stuff, build and
     install. This would need write access to python installation dirs:

  python setup.py install

  2. You can build and install in two separate commands

  python setup.py build
  python setup.py install

  3. Specific paths would be used prior to any automatic detection:

  python setup.py build --includes=/usr/local/includes:/home/tools/include

  4. Installation to a local directory (usual setup pattern)

  python setup.py install --prefix=/home/myself/install

"""


def str2bool(value):
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


pr = argparse.ArgumentParser(
    description=doc1,
    epilog=doc2,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    usage="python %(prog)s [options] file1 file2 ...",
)

pr.add_argument(
    "-I",
    "--includes",
    dest="incs",
    help="list of paths for include search ( : separated), order is significant and is kept unchanged",
)
pr.add_argument(
    "-L",
    "--libraries",
    dest="libs",
    help="list of paths for libraries search ( : separated), order is significant and is kept unchanged",
)
pr.add_argument("-U", "--update", action="store_true", help="update version (dev only)")
pr.add_argument(
    "-F", "--force", action="store_false", help="skip all .pyx files regeneration"
)
pr.add_argument(
    "-A",
    "--alternate",
    action="store_true",
    help="use full h5py CGNS/HDF5 interface (ongoing work)",
)

modules = {"app": True, "map": True, "pat": True, "val": True, "nav": True}

for name, val in modules.items():
    pr.add_argument(
        "--" + name,
        type=str2bool,
        default=val,
        help="enable/disable building of CGNS." + name.upper(),
    )

for hstr in ["--help", "-h", "help"]:
    if hstr in sys.argv:
        pr.print_help()
        sys.exit(1)

# Remove modules from command-line arguments
pr1 = argparse.ArgumentParser()
for name, val in modules.items():
    pr1.add_argument("--" + name, type=str2bool, default=val)

args1, unknown = pr1.parse_known_args()
sys.argv = sys.argv[:1] + unknown

if "help" in unknown:
    pr.print_help()
    sys.exit(1)

try:
    os.makedirs("./build/lib/CGNS")
except OSError:
    pass

log("using Python from {}".format(sys.prefix))

if HAS_MSW:
    log("found Windows platform")
else:
    log("found Unix platform")

APP = args1.app
MAP = args1.map
PAT = args1.pat
VAL = args1.val
NAV = args1.nav

ALL_PACKAGES = []
ALL_SCRIPTS = []
ALL_EXTENSIONS = []
OTHER_INCLUDES_PATHS = []
OTHER_LIBRARIES_PATHS = []
EXTRA_DEFINE_MACROS = []

if HAS_MSW:
    EXTRA_DEFINE_MACROS = [("_HDF5USEDLL_", None), ("H5_BUILT_AS_DYNAMIC_LIB", None)]

module_logs = []
incs = []
libs = []

args, unknown = pr.parse_known_args()

if "install" in unknown:
    args.force = False

if args.incs is not None:
    incs = [os.path.expanduser(path) for path in args.incs.split(os.path.pathsep)]
if args.libs is not None:
    libs = [os.path.expanduser(path) for path in args.libs.split(os.path.pathsep)]

if args.alternate:
    deps = ["Cython", "h5py", "numpy", "vtk", "qtpy"]
else:
    deps = ["Cython", "HDF5", "numpy", "vtk", "qtpy"]

RUN_REQUIRES = ["numpy", "future"]
SETUP_REQUIRES = RUN_REQUIRES + ["cython>=0.25", "pkgconfig"]

# Remove crashing deps test
if "sdist" in sys.argv:
    deps = ["Cython", "numpy", "vtk", "qtpy"]

# Dirty patch
# Get required EGG if needed
import setuptools.dist

dist = setuptools.dist.Distribution(dict({"setup_requires": SETUP_REQUIRES}))
try:
    (CONFIG, status) = search(incs, libs, deps=deps)
except ConfigException as e:
    log("***** Cannot build pyCGNS without: {}".format(e))
    sys.exit(1)

# Fake HDF5 Config
if "sdist" in sys.argv:
    CONFIG.HDF5_HST = 1
    CONFIG.HDF5_H64 = 1
    CONFIG.HDF5_HUP = 1
    CONFIG.HDF5_VERSION = "1.10.4"
    CONFIG.HDF5_PARALLEL = 0
    CONFIG.USE_COMPACT_STORAGE = 1


line()

new_args = []

for arg in sys.argv:
    if (
        not ("-I=" in arg or "--includes=" in arg)
        and not ("-U" in arg or "--update" in arg)
        and not ("-A" in arg or "--alternate" in arg)
        and not ("-F" in arg or "--force" in arg)
        and not ("-L=" in arg or "--libraries=" in arg)
    ):
        new_args += [arg]
sys.argv = new_args

if args.update:
    # os.system('hg parents --template="{rev}\n" > %s/revision.tmp' \
    #          % CONFIG.PRODUCTION_DIR)
    # Quick and dirty revision, should use git describe --always instead
    os.system(
        r"git rev-list --count HEAD> {}/revision.tmp".format(CONFIG.PRODUCTION_DIR)
    )
    updateVersionInFile(fix_path("./lib/pyCGNSconfig_default.py"), CONFIG)


def hasToGenerate(source, destination, force):
    return (
        force
        or not os.path.exists(destination)
        or os.path.getmtime(source) > os.path.getmtime(destination)
    )


def resolveVars(filename, confvalues, force):
    if hasToGenerate(filename + ".in", filename, force):
        with open("{}.in".format(filename), "r") as fg1:
            l1 = fg1.readlines()
        l2 = [ll.format(**confvalues) for ll in l1]
        with open(filename, "w") as fg2:
            fg2.writelines(l2)


if args.force:
    print("Generation of pyx not skipped")
else:
    print("Skipping pyx generation")

# -------------------------------------------------------------------------
if APP:
    slist = [
        "cg_grep",
        "cg_list",
        "cg_link",
        "cg_iges",
        "cg_diff",
        "cg_checksum",
        "cg_gather",
        "cg_scatter",
        "cg_dump",
        "cg_scan",
    ]
    if NAV:
        slist += ["cg_look"]

    ALL_SCRIPTS += ["CGNS/APP/tools/%s" % f for f in slist]

    ALL_EXTENSIONS += [
        Extension(
            "CGNS.APP.lib.arrayutils",
            ["CGNS/APP/lib/arrayutils.pyx", "CGNS/APP/lib/hashutils.c"],
            include_dirs=CONFIG.INCLUDE_DIRS
            + OTHER_INCLUDES_PATHS
            + [
                "CGNS/APP/lib",
            ],
            extra_compile_args=[],
        )
    ]
    ALL_PACKAGES += [
        "CGNS.APP",
        "CGNS.APP.lib",
        "CGNS.APP.tools",
        "CGNS.APP.examples",
        "CGNS.APP.misc",
        "CGNS.APP.test",
    ]
    module_logs.append("APP   add  build")
else:
    module_logs.append("APP   skip build *")

# -------------------------------------------------------------------------
if MAP:
    if not CONFIG.HAS_H5PY:
        # generate files
        # CHLone_config.h.in -> CHLone_config.h
        # pyCHLone.pyx.in -> pyCHLone.pyx
        #
        # --- config values
        hdfplib = CONFIG.HDF5_PATH_LIBRARIES
        hdflib = CONFIG.HDF5_LINK_LIBRARIES
        hdfpinc = CONFIG.HDF5_PATH_INCLUDES
        include_dirs = ["."] + hdfpinc + CONFIG.INCLUDE_DIRS + OTHER_INCLUDES_PATHS
        library_dirs = hdfplib
        optional_libs = hdflib
        extra_compile_args = CONFIG.HDF5_EXTRA_ARGS
        extra_define_macro = EXTRA_DEFINE_MACROS

        conf = {
            "CHLONE_HAS_PTHREAD": 1,
            "CHLONE_HAS_REGEXP": 1,
            "CHLONE_PRINTF_TRACE": 0,
            "CHLONE_ON_WINDOWS": HAS_MSW,
            "CHLONE_H5CONF_STD": CONFIG.HDF5_HST,
            "CHLONE_H5CONF_64": CONFIG.HDF5_H64,
            "CHLONE_H5CONF_UP": CONFIG.HDF5_HUP,
            "CHLONE_USE_COMPACT_STORAGE": CONFIG.USE_COMPACT_STORAGE,
            "HDF5_VERSION": CONFIG.HDF5_VERSION,
            "CHLONE_INSTALL_LIBRARIES": "",
            "CHLONE_INSTALL_INCLUDES": "",
        }

        depfiles = ["CGNS/MAP/CHLone_config.h", "CGNS/MAP/EmbeddedCHLone.pyx"]

        EXTRA_MAP_COMPILE_ARGS = ""

        resolveVars(fix_path(depfiles[0]), conf, args.force)
        resolveVars(fix_path(depfiles[1]), conf, args.force)
        library_dirs = [l for l in library_dirs if l != ""]

        # hack: actually shoudl read hdf5/cmake config to get true compiler...
        if CONFIG.HDF5_PARALLEL:
            os.environ["CC"] = CYTHON_COMPILER_FOR_MAP

        ALL_EXTENSIONS += [
            Extension(
                "CGNS.MAP.EmbeddedCHLone",
                [
                    "CGNS/MAP/EmbeddedCHLone.pyx",
                    "CGNS/MAP/SIDStoPython.c",
                    "CGNS/MAP/l3.c",
                    "CGNS/MAP/error.c",
                    "CGNS/MAP/linksearch.c",
                    "CGNS/MAP/sha256.c",
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=optional_libs,
                depends=depfiles,
                define_macros=extra_define_macro,
                extra_compile_args=extra_compile_args,
            )
        ]

    ALL_PACKAGES += ["CGNS.MAP", "CGNS.MAP.test"]
    module_logs.append("MAP   add  build")
else:
    module_logs.append("MAP   skip build *")

# -------------------------------------------------------------------------
if VAL:
    ALL_PACKAGES += [
        "CGNS.VAL",
        "CGNS.VAL.grammars",
        "CGNS.VAL.suite",
        "CGNS.VAL.suite.SIDS",
        "CGNS.VAL.parse",
        "CGNS.VAL.test",
    ]
    ALL_SCRIPTS += ["CGNS/VAL/CGNS.VAL"]

    if args.force:
        touch("CGNS/VAL/grammars/CGNS_VAL_USER_SIDS_.pyx")
        touch("CGNS/VAL/grammars/etablesids.pyx")
        touch("CGNS/VAL/grammars/valutils.pyx")

    ALL_EXTENSIONS += [
        Extension(
            "CGNS.VAL.grammars.CGNS_VAL_USER_SIDS_",
            ["CGNS/VAL/grammars/CGNS_VAL_USER_SIDS_.pyx"],
            include_dirs=CONFIG.INCLUDE_DIRS,
            extra_compile_args=[],
        )
    ]
    ALL_EXTENSIONS += [
        Extension(
            "CGNS.VAL.grammars.etablesids",
            ["CGNS/VAL/grammars/etablesids.pyx"],
            include_dirs=CONFIG.INCLUDE_DIRS,
            extra_compile_args=[],
        )
    ]
    ALL_EXTENSIONS += [
        Extension(
            "CGNS.VAL.grammars.valutils",
            ["CGNS/VAL/grammars/valutils.pyx"],
            include_dirs=CONFIG.INCLUDE_DIRS,
            extra_compile_args=[],
        )
    ]

    module_logs.append("VAL   add  build")
else:
    module_logs.append("VAL   skip build *")

# -------------------------------------------------------------------------
if PAT:
    # if CONFIG.HAS_CYTHON:
    #  ALL_EXTENSIONS+=[ Extension('CGNS.PAT.cgnsutils',
    #                              ['CGNS/PAT/cgnsutils.pyx'],
    #                              include_dirs = CONFIG.INCLUDE_DIRS) ]
    ALL_PACKAGES += ["CGNS.PAT", "CGNS.PAT.SIDS", "CGNS.PAT.test"]
    module_logs.append("PAT   add  build")
else:
    module_logs.append("PAT   skip build *")

# -------------------------------------------------------------------------
if NAV and CONFIG.HAS_QTPY:
    cui = CONFIG.COM_UIC
    crc = CONFIG.COM_RCC
    ccy = CONFIG.COM_CYTHON

    fakefile = "./CGNS/NAV/fake.pxi"
    if args.force:
        touch(fakefile)

    modnamelist = [
        "Q7TreeWindow",
        "Q7DiffWindow",
        "Q7MergeWindow",
        "Q7ControlWindow",
        "Q7OptionsWindow",
        "Q7FormWindow",
        "Q7FileWindow",
        "Q7QueryWindow",
        "Q7SelectionWindow",
        "Q7InfoWindow",
        "Q7DiagWindow",
        "Q7LinkWindow",
        "Q7HelpWindow",
        "Q7ToolsWindow",
        "Q7PatternWindow",
        "Q7AnimationWindow",
        "Q7MessageWindow",
        "Q7LogWindow",
    ]
    if CONFIG.HAS_VTK:
        modnamelist += ["Q7VTKWindow"]
    modgenlist = []
    modextlist = []
    mfile_list = [
        "mtree",
        "mquery",
        "mcontrol",
        "mtable",
        "mpattern",
        "diff",
        "mdifftreeview",
        "merge",
        "mmergetreeview",
    ]
    if CONFIG.HAS_VTK:
        mfile_list += ["mparser"]

    for mfile in mfile_list:
        if args.force:
            touch("CGNS/NAV/%s.pyx" % mfile)
        modextlist += [
            Extension(
                "CGNS.NAV.%s" % mfile,
                ["CGNS/NAV/%s.pyx" % mfile],
                include_dirs=CONFIG.NUMPY_PATH_INCLUDES,
                library_dirs=CONFIG.NUMPY_PATH_LIBRARIES,
                libraries=CONFIG.NUMPY_LINK_LIBRARIES,
            )
        ]
    for m in modnamelist:
        modextlist += [
            Extension(
                "CGNS.NAV.%s" % m,
                ["CGNS/NAV/G/%s.pyx" % m],
                include_dirs=CONFIG.NUMPY_PATH_INCLUDES,
                library_dirs=CONFIG.NUMPY_PATH_LIBRARIES,
                libraries=CONFIG.NUMPY_LINK_LIBRARIES,
            )
        ]
        g = ("CGNS/NAV/T/{}.ui".format(m), "CGNS/NAV/G/{}.pyx".format(m))
        if ("true" not in [cui, crc]) and hasToGenerate(g[0], g[1], args.force):
            modgenlist += [m]

    for m in modgenlist:
        log("Generate from updated Qt templates  ({}): {}".format(cui, m))
        com = "{} --from-imports -o CGNS/NAV/G/{}.pyx CGNS/NAV/T/{}.ui".format(
            cui, m, m
        )
        os.system(fix_path(com))
        com = "{} -X language_level=3 -a CGNS/NAV/G/{}.pyx".format(ccy, m)
        os.system(fix_path(com))

    if hasToGenerate("CGNS/NAV/R/Res.qrc", "CGNS/NAV/Res_rc.py", args.force):
        log("Generate from updated Qt Ressources ({}): Res_rc.py".format(crc))
        com = "{} -o CGNS/NAV/Res_rc.py CGNS/NAV/R/Res.qrc".format(crc)
        os.system(fix_path(com))

    ALL_PACKAGES += ["CGNS.NAV", "CGNS.NAV.test"]
    ALL_EXTENSIONS += modextlist

    if CONFIG.HAS_VTK:
        module_logs.append("NAV   add  build (with VTK)")
    else:
        module_logs.append("NAV   add  build (without VTK)")

else:
    module_logs.append("NAV   skip build *")

installConfigFiles(CONFIG.PRODUCTION_DIR)


for e in ALL_EXTENSIONS:
    e.cython_directives = {"language_level": "3"}

#  -------------------------------------------------------------------------
if CONFIG.HAS_CYTHON:
    from Cython.Distutils import build_ext

    cmd = {"clean": clean, "build_ext": build_ext}
else:
    cmd = {"clean": clean}

for module_log in module_logs:
    log(module_log)
line()

# --- Distutils metadata --------------------------------------------

cls_txt = """
Development Status :: 3 - Alpha
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)
Programming Language :: Cython
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Database
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

# -------------------------------------------------------------------------
long_description = (Path(__file__).parent / "README.md").read_text()

# -------------------------------------------------------------------------
setup(
    name=CONFIG.NAME,
    version="{}.{}.{}".format(MAJOR_VERSION, MINOR_VERSION, CONFIG.REVISION),
    description=CONFIG.DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[x for x in cls_txt.split("\n") if x],
    author=CONFIG.AUTHOR,
    author_email=CONFIG.EMAIL,
    license=CONFIG.LICENSE,
    url=CONFIG.URL,
    packages=ALL_PACKAGES,
    scripts=ALL_SCRIPTS,
    ext_modules=ALL_EXTENSIONS,
    cmdclass=cmd,
    install_requires=RUN_REQUIRES,
    setup_requires=SETUP_REQUIRES,
)
# -------------------------------------------------------------------------
# --- last line
