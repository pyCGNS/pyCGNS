#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from __future__ import print_function
import os
import sys
import string
import argparse
import glob
import re

from setuptools import setup, Extension

# --- get overall configuration
sys.path = ['./lib'] + sys.path
import setuputils


def line(m=""):
    print("#", "-" * 70)
    if m:
        print("# ----- %s" % m)


line('pyCGNS v%d.%d install' % (setuputils.MAJORVERSION, setuputils.MINORVERSION))
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
  http://pycgns.sourceforge.net/install.html

  MPI: using HDF5 with parallel features adds dependancies on mpi. The
  simple way to resolve these deps is to force mpicc instead of cc:

  CC=mpicc python setup.py build

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

pr = argparse.ArgumentParser(description=doc1, epilog=doc2,
                             formatter_class=argparse.RawDescriptionHelpFormatter,
                             usage='python %(prog)s [options] file1 file2 ...')

pr.add_argument("-I", "--includes", dest="incs",
                help='list of paths for include search ( : separated), order is significant and is kept unchanged')
pr.add_argument("-L", "--libraries", dest="libs",
                help='list of paths for libraries search ( : separated), order is significant and is kept unchanged')
pr.add_argument("-U", "--update", action='store_true',
                help='update version (dev only)')
pr.add_argument("-g", "--generate", action='store_true',
                help='force Qt/creator .pyx files to be regenerated')

try:
    os.makedirs('./build/lib/CGNS')
except OSError:
    pass

APP = True
MAP = True
WRA = False
MAP = True
PAT = True
VAL = True
DAT = True
NAV = True

ALL_PACKAGES = []
ALL_SCRIPTS = []
ALL_EXTENSIONS = []
OTHER_INCLUDES_PATHS = []
OTHER_LIBRARIES_PATHS = []

if sys.platform == 'win32':
    OTHER_INCLUDES_PATHS = ['c:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\include'] + [
        'c:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.1A\\include']
    OTHER_INCLUDES_PATHS += ['.\\windows']
    OTHER_LIBRARIES_PATHS = ['c:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\lib\\amd64'] + [
        'c:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.1A\\Lib\\x64']
    EXTRA_DEFINE_MACROS = [('_HDF5USEDLL_', None), ('H5_BUILT_AS_DYNAMIC_LIB', None)]
    PLATFORM_WINDOWS = 1
else:
    EXTRA_DEFINE_MACROS = []
    PLATFORM_WINDOWS = 0

modules = ""

incs = []
libs = []

args, unknown = pr.parse_known_args()

if args.incs is not None:
    incs = [os.path.expanduser(path) for path in args.incs.split(os.path.pathsep)]
if args.libs is not None:
    libs = [os.path.expanduser(path) for path in args.libs.split(os.path.pathsep)]

try:
    (CONFIG, status) = setuputils.search(incs, libs)
except setuputils.ConfigException as val:
    print('Cannot build pyCGNS without:', val)
    sys.exit(1)

print(setuputils.pfx + '-' * 65)

new_args = []
for arg in sys.argv:
    if (not ('-I=' in arg or '--includes=' in arg) and
            not ('-U' in arg or '--update' in arg) and
            not ('-g' in arg or '--generate' in arg) and
            not ('-L=' in arg or '--libraries=' in arg)):
        new_args += [arg]
sys.argv = new_args

if args.update:
    os.system('hg parents --template="{rev}\n" > %s/revision.tmp' \
              % CONFIG.PRODUCTION_DIR)
    setuputils.updateVersionInFile('./lib/pyCGNSconfig_default.py',
                                   CONFIG.PRODUCTION_DIR)


def hasToGenerate(source, destination, force):
    if PLATFORM_WINDOWS:
        return True
    return (force or not os.path.exists(destination) or
                os.path.getmtime(source) > os.path.getmtime(destination))


def resolveVars(filename, conf, force):
    if hasToGenerate(filename + '.in', filename, args.generate):
        with open('%s.in' % filename, 'r') as fg1:
            l1 = fg1.readlines()
        l2 = [ll % conf for ll in l1]
        with open(filename, 'w') as fg2:
            fg2.writelines(l2)


# -------------------------------------------------------------------------
if APP:
    slist = ['cg_grep', 'cg_list', 'cg_link', 'cg_iges', 'cg_diff', 'cg_checksum',
             'cg_gather', 'cg_scatter', 'cg_dump',
             'cg_scan', 'cg_look']

    ALL_SCRIPTS += ['CGNS/APP/tools/%s' % f for f in slist]

    ALL_EXTENSIONS += [Extension("CGNS.APP.lib.arrayutils",
                                 ["CGNS/APP/lib/arrayutils.pyx",
                                  "CGNS/APP/lib/hashutils.c"],
                                 include_dirs=CONFIG.INCLUDE_DIRS + OTHER_INCLUDES_PATHS,
                                 extra_compile_args=[])]
    ALL_PACKAGES += ['CGNS.APP',
                     'CGNS.APP.lib',
                     'CGNS.APP.tools',
                     'CGNS.APP.examples',
                     'CGNS.APP.misc',
                     'CGNS.APP.test']
    modules += "\n# APP   add  build"
else:
    modules += "\n# APP   skip build *"

# -------------------------------------------------------------------------  
if MAP:
    # generate files
    # CHLone_config.h.in -> CHLone_config.h
    # pyCHLone.pyx.in -> pyCHLone.pyx
    #
    # --- config values
    hdfplib = CONFIG.HDF5_PATH_LIBRARIES
    hdflib = CONFIG.HDF5_LINK_LIBRARIES
    hdfpinc = CONFIG.HDF5_PATH_INCLUDES
    include_dirs = ['.'] + hdfpinc + CONFIG.INCLUDE_DIRS + OTHER_INCLUDES_PATHS
    library_dirs = hdfplib
    optional_libs = hdflib
    extra_compile_args = CONFIG.HDF5_EXTRA_ARGS
    extra_define_macro = EXTRA_DEFINE_MACROS

    conf = {'CHLONE_HAS_PTHREAD': 1,
            'CHLONE_HAS_REGEXP': 1,
            'CHLONE_PRINTF_TRACE': 0,
            'CHLONE_ON_WINDOWS': PLATFORM_WINDOWS,
            'CHLONE_H5CONF_STD': CONFIG.HDF5_HST,
            'CHLONE_H5CONF_64': CONFIG.HDF5_H64,
            'CHLONE_H5CONF_UP': CONFIG.HDF5_HUP,
            'HDF5_VERSION': CONFIG.HDF5_VERSION,
            'CHLONE_INSTALL_LIBRARIES': "",
            'CHLONE_INSTALL_INCLUDES': "",
            }

    depfiles = ['CGNS/MAP/CHLone_config.h', 'CGNS/MAP/EmbeddedCHLone.pyx']

    EXTRA_MAP_COMPILE_ARGS = ''

    for d in depfiles: resolveVars(d, conf, args.generate)
    library_dirs = [l for l in library_dirs if l != '']

    ALL_EXTENSIONS += [Extension("CGNS.MAP.EmbeddedCHLone",
                                 ["CGNS/MAP/EmbeddedCHLone.pyx",
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
                                 extra_compile_args=extra_compile_args)]

    ALL_PACKAGES += ['CGNS.MAP', 'CGNS.MAP.test']
    modules += "\n# MAP   add  build"
else:
    modules += "\n# MAP   skip build *"

# -------------------------------------------------------------------------  
if VAL:
    ALL_PACKAGES += ['CGNS.VAL',
                     'CGNS.VAL.grammars',
                     'CGNS.VAL.suite',
                     'CGNS.VAL.suite.SIDS',
                     'CGNS.VAL.parse',
                     'CGNS.VAL.test']
    ALL_SCRIPTS += ['CGNS/VAL/CGNS.VAL']

    ALL_EXTENSIONS += [Extension("CGNS.VAL.grammars.CGNS_VAL_USER_SIDS_",
                                 ["CGNS/VAL/grammars/CGNS_VAL_USER_SIDS_.pyx"],
                                 include_dirs=CONFIG.INCLUDE_DIRS,
                                 extra_compile_args=[])]
    ALL_EXTENSIONS += [Extension("CGNS.VAL.grammars.etablesids",
                                 ["CGNS/VAL/grammars/etablesids.pyx"],
                                 include_dirs=CONFIG.INCLUDE_DIRS,
                                 extra_compile_args=[])]
    ALL_EXTENSIONS += [Extension("CGNS.VAL.grammars.valutils",
                                 ["CGNS/VAL/grammars/valutils.pyx"],
                                 include_dirs=CONFIG.INCLUDE_DIRS,
                                 extra_compile_args=[])]

    modules += "\n# VAL   add  build"
else:
    modules += "\n# VAL   skip build *"

# -------------------------------------------------------------------------  
if PAT:
    # if CONFIG.HAS_CYTHON:
    #  ALL_EXTENSIONS+=[ Extension('CGNS.PAT.cgnsutils',
    #                              ['CGNS/PAT/cgnsutils.pyx'],
    #                              include_dirs = CONFIG.INCLUDE_DIRS) ]
    ALL_PACKAGES += ['CGNS.PAT',
                     'CGNS.PAT.SIDS',
                     'CGNS.PAT.test']
    modules += "\n# PAT   add  build"
else:
    modules += "\n# PAT   skip build *"

# -------------------------------------------------------------------------  
if (WRA and CONFIG.HAS_MLL and CONFIG.HAS_CYTHON_2PLUS):

    # --- config values
    hdfplib = CONFIG.HDF5_PATH_LIBRARIES
    hdflib = CONFIG.HDF5_LINK_LIBRARIES
    hdfpinc = CONFIG.HDF5_PATH_INCLUDES
    hdfversion = CONFIG.HDF5_VERSION
    mllplib = CONFIG.MLL_PATH_LIBRARIES
    mlllib = CONFIG.MLL_LINK_LIBRARIES
    mllpinc = CONFIG.MLL_PATH_INCLUDES
    mllversion = CONFIG.MLL_VERSION

    lname = "CGNS.WRA"

    extraargs = CONFIG.MLL_EXTRA_ARGS
    include_dirs = CONFIG.INCLUDE_DIRS + ['WRA/modadf']
    library_dirs = CONFIG.LIBRARY_DIRS
    optional_libs = mlllib + hdflib

    ALL_PACKAGES += ['CGNS.WRA', 'CGNS.WRA.test']

    if CONFIG.HAS_CYTHON:
        ALL_EXTENSIONS += [Extension('CGNS.WRA.mll', ['CGNS/WRA/mll.pyx',
                                                      'CGNS/WRA/mll_utils.c'],
                                     include_dirs=include_dirs + ['WRA'],
                                     library_dirs=library_dirs,
                                     libraries=optional_libs,
                                     extra_compile_args=extraargs)]

    modules += "\n# WRA   add  build"
else:
    modules += "\n# WRA   skip build *"

# -------------------------------------------------------------------------  
if DAT:
    ALL_PACKAGES += ['CGNS.DAT',
                     'CGNS.DAT.db',
                     'CGNS.DAT.db.dbdrivers',
                     'CGNS.DAT.demo',
                     'CGNS.DAT.test']
    ALL_SCRIPTS += ['CGNS/DAT/tools/CGNS.DAT',
                    'CGNS/DAT/tools/daxQT',
                    'CGNS/DAT/tools/CGNS.DAT.create']
    modules += "\n# DAT   add  build"
else:
    modules += "\n# DAT   skip build *"

# -------------------------------------------------------------------------  
if NAV and CONFIG.HAS_QTPY:
    cui = CONFIG.COM_UIC
    crc = CONFIG.COM_RCC
    ccy = CONFIG.COM_CYTHON

    fakefile = "./CGNS/NAV/fake.pxi"
    if args.generate:
        setuputils.touch(fakefile)

    modnamelist = [
        'Q7TreeWindow',
        'Q7DiffWindow',
        'Q7MergeWindow',
        'Q7ControlWindow',
        'Q7OptionsWindow',
        'Q7FormWindow',
        'Q7FileWindow',
        'Q7QueryWindow',
        'Q7SelectionWindow',
        'Q7InfoWindow',
        'Q7DiagWindow',
        'Q7LinkWindow',
        'Q7HelpWindow',
        'Q7ToolsWindow',
        'Q7PatternWindow',
        'Q7AnimationWindow',
        'Q7MessageWindow',
        'Q7LogWindow',
    ]
    if CONFIG.HAS_VTK:
        modnamelist += ['Q7VTKWindow']
    modgenlist = []
    modextlist = []
    mfile_list = ['mtree', 'mquery', 'mcontrol', 'mtable', 'mpattern',
                  'diff', 'mdifftreeview', 'merge', 'mmergetreeview']
    if CONFIG.HAS_VTK:
        mfile_list += ['mparser']

    for mfile in mfile_list:
        modextlist += [Extension("CGNS.NAV.%s" % mfile, ["CGNS/NAV/%s.pyx" % mfile,
                                                         fakefile],
                                 include_dirs=CONFIG.NUMPY_PATH_INCLUDES,
                                 library_dirs=CONFIG.NUMPY_PATH_LIBRARIES,
                                 libraries=CONFIG.NUMPY_LINK_LIBRARIES,
                                 )]
    for m in modnamelist:
        modextlist += [Extension("CGNS.NAV.%s" % m, ["CGNS/NAV/G/%s.pyx" % m,
                                                     fakefile],
                                 include_dirs=CONFIG.NUMPY_PATH_INCLUDES,
                                 library_dirs=CONFIG.NUMPY_PATH_LIBRARIES,
                                 libraries=CONFIG.NUMPY_LINK_LIBRARIES,
                                 )]
        g = ("CGNS/NAV/T/%s.ui" % m, "CGNS/NAV/G/%s.pyx" % m)
        if (('true' not in [cui, crc]) and hasToGenerate(g[0], g[1], args.generate)):
            modgenlist += [m]

    for m in modgenlist:
        print('# Generate from updated Qt templates  (%s): %s' % (cui, m))
        com = "(%s -o CGNS/NAV/G/%s.pyx CGNS/NAV/T/%s.ui;(cd CGNS/NAV/G;%s -a %s.pyx))2>/dev/null" % (cui, m, m, ccy, m)
        # print com
        os.system(com)

    if (hasToGenerate('CGNS/NAV/R/Res.qrc', 'CGNS/NAV/Res_rc.py', args.generate)):
        print('# Generate from updated Qt Ressources (%s): Res_rc.py' % (crc))
        com = "(%s -o CGNS/NAV/Res_rc.py CGNS/NAV/R/Res.qrc)2>/dev/null" % (crc)
        os.system(com)

    ALL_PACKAGES += ['CGNS.NAV', 'CGNS.NAV.test']
    ALL_EXTENSIONS += modextlist

    modules += "\n# NAV   add  build"
else:
    modules += "\n# NAV   skip build *"

setuputils.installConfigFiles(CONFIG.PRODUCTION_DIR)

#  -------------------------------------------------------------------------
if CONFIG.HAS_CYTHON:
    from Cython.Distutils import build_ext

    cmd = {'clean': setuputils.clean, 'build_ext': build_ext}
else:
    cmd = {'clean': setuputils.clean}

print("#" + modules)
print("#\n# Running build now...\n#")

# -------------------------------------------------------------------------  
setup(
    name=CONFIG.NAME,
    version=CONFIG.VERSION,
    description=CONFIG.DESCRIPTION,
    author=CONFIG.AUTHOR,
    author_email=CONFIG.EMAIL,
    license=CONFIG.LICENSE,
    packages=ALL_PACKAGES,
    scripts=ALL_SCRIPTS,
    ext_modules=ALL_EXTENSIONS,
    cmdclass=cmd, install_requires=['numpy']
)
# -------------------------------------------------------------------------  
# --- last line
