#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# --------------------------------------------------------------------
MAJOR_VERSION = 6
MINOR_VERSION = 0
REVISION = 0
# --------------------------------------------------------------------

import os
import platform
import sys
import shutil
import re
import time
import pathlib
import distutils.util

from distutils.dir_util import remove_tree
from distutils.command.clean import clean as _clean

rootfiles = ['__init__.py', 'errors.py', 'version.py', 'config.py', 'test.py']
compfiles = []

pfx = '#'
def line(msg=""):
    print("{} {}".format(pfx, "-" * 70))
    if msg:
        print("{} --- {}".format(pfx, msg))

def log(msg):
    print("{} {}".format(pfx, msg))

# if you change this name, also change lines tagged with 'USER CONFIG'
userconfigfile = 'setup_userConfig.py'

class ConfigException(Exception):
    pass

def is_windows():
    if sys.platform == 'win32':
        return 1
    else:
        return 0

def is_python3():
    if sys.version_info[0] < 3:
        return 0
    else:
        return 1


# Please leave integers here, these will be used in the SIDS-to-Python C code
HAS_MSW = is_windows()

def fix_path(path):
    """All paths should be POSIX paths. Translation is required only for windows."""
    return pathlib.Path(path).as_posix()
        
# --------------------------------------------------------------------
def prodtag():
    from time import gmtime, strftime
    proddate = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    try:
        prodhost = platform.uname()
    except AttributeError:
        prodhost = ('???', '???', '???')
    return (proddate, prodhost)

# http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
            w_exe_file = exe_file + '.exe'
            if is_exe(w_exe_file):
                return w_exe_file
            b_exe_file = exe_file + '.bat'
            if is_exe(b_exe_file):
                return b_exe_file

    return None

# --------------------------------------------------------------------
def unique_but_keep_order(lst):
    if len(lst) < 2:
        return lst
    r = [lst[0]]
    for p in lst[1:]:
        if p not in r:
            r.append(p)
    return r

# --------------------------------------------------------------------
def search(incs, libs, tag='pyCGNS',
           deps=['Cython', 'HDF5', 'numpy', 'vtk', 'qtpy']):
    state = 1
    for com in sys.argv:
        if com in ['help', 'clean']: state = 0
    pt = distutils.util.get_platform()
    vv = "%d.%d" % (sys.version_info[0], sys.version_info[1])
    tg = "%s/./build/lib.%s-%s/CGNS" % (os.getcwd(), pt, vv)
    bptarget = tg
    if (not os.path.exists(bptarget)): os.makedirs(bptarget)
    oldsyspath = sys.path
    sys.path = [os.path.abspath(os.path.normpath('./lib'))]
    cfgdict = {}
    import pyCGNSconfig_default as C_D
    sys.path = oldsyspath
    for ck in dir(C_D):
        if ck[0] != '_':
            cfgdict[ck] = C_D.__dict__[ck]
    pg = prodtag()
    cfgdict['PFX'] = pfx
    cfgdict['DATE'] = pg[0]
    cfgdict['PLATFORM'] = "%s %s %s" % (pg[1][0], pg[1][1], pg[1][-1])
    cfgdict['HAS_MSW'] = HAS_MSW
    updateConfig('..', bptarget, cfgdict)
    sys.path = [bptarget] + sys.path

    # here we go, check each dep and add incs/libs/others to config
    try:
        import pyCGNSconfig as C

        # -----------------------------------------------------------------------
        C.COM_UIC = 'true'
        C.COM_RCC = 'true'
        C.COM_CYTHON = 'true'        
        if ('Cython' in deps):
            try:
                if (which('cython') is not None):
                    C.COM_CYTHON = 'cython'
                elif (which('cython3') is not None):
                    C.COM_CYTHON = 'cython3'
                else:
                    raise Exception
                if (which('pyuic') is not None):  C.COM_UIC = 'pyuic'
                if (which('pyrcc') is not None):  C.COM_RCC = 'pyrcc'
                if (which('pyuic5') is not None): C.COM_UIC = 'pyuic5'
                if (which('pyrcc5') is not None): C.COM_RCC = 'pyrcc5'
                import Cython
                C.HAS_CYTHON = True
                log('found Cython v%s' % Cython.__version__)
                log('using Cython from {}'.format(os.path.dirname(Cython.__file__)))
                C.HAS_CYTHON_2PLUS = False
                C.CYTHON_VERSION = Cython.__version__
                try:
                    if (float(Cython.__version__[:3]) > 0.1):
                        C.HAS_CYTHON_2PLUS = True
                    else:
                        log('***** SKIP Cython version cannot build CGNS')
                except:
                    log('***** SKIP Cython version cannot build CGNS')
            except:
                C.HAS_CYTHON = False
                log('***** FATAL: Cython not found')

        # -----------------------------------------------------------------------
        if ('qtpy' in deps):
            try:
                import qtpy
                import qtpy.QtCore
                import qtpy.QtGui
                import qtpy.QtWidgets

                C.HAS_QTPY = True
                log('found qtpy v{}'.format(qtpy.__version__))
                log('using qtpy from {}'.format(os.path.dirname(qtpy.__file__)))
                log('using qtpy with Qt v{}'.format(qtpy.QtCore.__version__))
                C.PYQT_VERSION = str(qtpy.__version__)
                C.QT_VERSION = str(qtpy.QtCore.__version__)
            except:
                C.HAS_QTPY = False
                log('***** SKIP NAV: qtpy not found')

        # -----------------------------------------------------------------------
        if ('vtk' in deps):
            try:
                import vtk
                v = vtk.vtkVersion()
                C.HAS_VTK = True
                log('found vtk (python module) v%s' % v.GetVTKVersion())
                C.VTK_VERSION = v.GetVTKVersion()
            except:
                C.HAS_VTK = False
                log('***** SKIP NAV/VTK: no vtk python module')

        # -----------------------------------------------------------------------
        if ('h5py' in deps):
            log('using h5py alternate implementation for CGNS.MAP')
            try:
                import warnings
                warnings.simplefilter(action='ignore', category=FutureWarning)
                import h5py
                v = h5py.version.version
                C.HDF5_VERSION = h5py.version.hdf5_version
                log('found h5py (python module) v{}'.format(h5py.version.version))
                log('using HDF5 v{}'.format(C.HDF5_VERSION))
                C.HAS_H5PY = 1
            except:
                log('***** FATAL: setup cannot find h5py and its HDF5')
                sys.exit(1)

        # -----------------------------------------------------------------------
        if ('numpy' in deps):
            incs = incs + C.NUMPY_PATH_INCLUDES
            libs = libs + C.NUMPY_PATH_LIBRARIES
            tp = find_numpy(incs, libs, C.NUMPY_LINK_LIBRARIES)
            if (tp is None):
                log('***** FATAL: setup cannot find Numpy')
                sys.exit(1)
            (C.NUMPY_VERSION,
             C.NUMPY_VERSION_API,
             C.NUMPY_PATH_INCLUDES,
             C.NUMPY_PATH_LIBRARIES,
             C.NUMPY_LINK_LIBRARIES,
             C.NUMPY_EXTRA_ARGS) = tp
            log('found Numpy version %s' % (C.NUMPY_VERSION,))
            log('using Numpy API version %s' % (C.NUMPY_VERSION_API,))
            log('using Numpy headers from %s' % (C.NUMPY_PATH_INCLUDES[0]))
            C.HAS_NUMPY = True
            incs = incs + C.NUMPY_PATH_INCLUDES
            libs = libs + C.NUMPY_PATH_LIBRARIES

        # -----------------------------------------------------------------------
        if ('HDF5' in deps):
            log('using HDF5 raw C API implementation for CGNS.MAP')
            try:
                import pkgconfig
            except ImportError:
                log('Failed to import pkgconfig')
                class pkgconfig(object):
                    @classmethod
                    def exists(cls, name):
                        return False
            try:
               if pkgconfig.exists('hdf5'):
                   log('Using pkgconfig for HDF5 detection')
                   pkgcfg = pkgconfig.parse("hdf5")
                   incs = incs + pkgcfg['include_dirs']
                   libs = libs + pkgcfg['library_dirs']
            except EnvironmentError:
                pass
            hdf5_ci = os.environ.get('HDF5_DIR')
            if hdf5_ci is not None:
               log('Using HDF5_DIR environment variable')
               incs = incs + [os.path.join(hdf5_ci, 'include'),]
               libs = libs + [os.path.join(hdf5_ci, 'lib'),]
            incs = incs + C.HDF5_PATH_INCLUDES + C.INCLUDE_DIRS
            libs = libs + C.HDF5_PATH_LIBRARIES + C.LIBRARY_DIRS
            tp = find_HDF5(incs, libs, C.HDF5_LINK_LIBRARIES)
            if (tp is None):
                log('***** FATAL: setup cannot find HDF5')
                sys.exit(1)
            (C.HDF5_VERSION,
             C.HDF5_PATH_INCLUDES,
             C.HDF5_PATH_LIBRARIES,
             C.HDF5_LINK_LIBRARIES,
             C.HDF5_EXTRA_ARGS,
             C.HDF5_HST,
             C.HDF5_H64,
             C.HDF5_HUP,
             C.HDF5_PARALLEL) = tp
            log('found HDF5 %s' % (C.HDF5_VERSION,))
            log('using HDF5 headers from %s' % (C.HDF5_PATH_INCLUDES[0]))
            log('using HDF5 libs from %s' % (C.HDF5_PATH_LIBRARIES[0]))
            if C.HDF5_PARALLEL:
                log('using HDF5 parallel version (cython uses $MPICC)')
            C.HAS_HDF5 = True
            C.HAS_H5PY = 0
            incs = incs + C.HDF5_PATH_INCLUDES + C.INCLUDE_DIRS
            libs = libs + C.HDF5_PATH_LIBRARIES + C.LIBRARY_DIRS

            # ---------------------------------------------------------------------

    except ImportError:
        log('***** FATAL: setup cannot find pyCGNSconfig.py file!')
        sys.exit(1)
    C.HDF5_PATH_INCLUDES = list(set(C.HDF5_PATH_INCLUDES))
    C.HDF5_PATH_LIBRARIES = list(set(C.HDF5_PATH_LIBRARIES))
    C.NUMPY_PATH_INCLUDES = list(set(C.NUMPY_PATH_INCLUDES))
    C.NUMPY_PATH_LIBRARIES = list(set(C.NUMPY_PATH_LIBRARIES))

    incs = unique_but_keep_order(incs)
    libs = unique_but_keep_order(libs)

    C.INCLUDE_DIRS = incs
    C.LIBRARY_DIRS = libs

    C.PRODUCTION_DIR = bptarget

    updateConfig('..', bptarget, C.__dict__, cfgdict)

    return (C, state)


# --------------------------------------------------------------------
def installConfigFiles(bptarget):
    lptarget = '.'
    for ff in rootfiles:
        shutil.copy("%s/lib/%s" % (lptarget, ff), "%s/%s" % (bptarget, ff))
    for ff in compfiles:
        shutil.copy("%s/lib/compatibility/%s" % (lptarget, ff), "%s/%s" % (bptarget, ff))


# --------------------------------------------------------------------
def updateVersionInFile(filename, cfg):
    f = open('{}/revision.tmp'.format(cfg.PRODUCTION_DIR))
    r = int(f.readlines()[0][:-1])
    REVISION = r
    f = open(filename, 'r')
    l = f.readlines()
    f.close()
    vver = '@@UPDATEVERSION@@'
    vrel = '@@UPDATERELEASE@@'
    vrev = '@@UPDATEREVISION@@'
    r = []
    for ll in l:
        rl = ll
        if (ll[-len(vver) - 1:-1] == vver):
            rl = '__version__=%s # %s\n' % (MAJOR_VERSION, vver)
        if (ll[-len(vrel) - 1:-1] == vrel):
            rl = '__release__=%s # %s\n' % (MINOR_VERSION, vrel)
        if (ll[-len(vrev) - 1:-1] == vrev):
            ACTUALREV = REVISION
            rl = '__revision__=%s # %s\n' % (ACTUALREV, vrev)
        r += [rl]
    f = open(filename, 'w+')
    f.writelines(r)
    f.close()
    cfg.REVISION = REVISION

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything
relist = [r'^.*~$', r'^core\.*$', r'^pyCGNS\.log\..*$',
          r'^#.*#$', r'^.*\.aux$', r'^.*\.pyc$', r'^.*\.bak$', r'^.*\.l2h',
          r'^Output.*$']
reclean = []

for restring in relist:
    reclean.append(re.compile(restring))


def wselect(args, dirname, names):
    for n in names:
        for rev in reclean:
            if (rev.match(n)):
                # print "%s/%s"%(dirname,n)
                os.remove("%s/%s" % (dirname, n))
                break


class clean(_clean):
    def run(self):
        import glob
        rdirs = glob.glob("./build/*")
        for d in rdirs: remove_tree(d)
        if os.path.exists("./build"):     remove_tree("./build")
        if os.path.exists("./Doc/_HTML"): remove_tree("./Doc/_HTML")
        if os.path.exists("./Doc/_PS"):   remove_tree("./Doc/_PS")
        if os.path.exists("./Doc/_PDF"):  remove_tree("./Doc/_PDF")


# --------------------------------------------------------------------
def confValueAsStr(v):
    if (type(v) == type((1,))): return str(v)
    if (type(v) == type([])):   return str(v)
    if (v in [True, False]):
        return str(v)
    else:
        return '"%s"' % str(v)


# --------------------------------------------------------------------
def updateConfig(pfile, gfile, config_default, config_previous=None):
    if (config_previous):
        from pyCGNSconfig_default import file_pattern as fpat
        cfg = config_default
        for ck in config_previous:
            if ck not in cfg:
                cfg[ck] = config_previous[ck]
        log("+++++ update pyCGNSconfig.py file")
        os.unlink("%s/pyCGNSconfig.py" % (gfile))
        f = open("%s/pyCGNSconfig.py" % (gfile), 'w+')
        f.writelines(fpat % cfg)
        f.close()
        return
    elif (not os.path.exists("%s/pyCGNSconfig.py" % (gfile))):
        log("+++++ create new pyCGNSconfig.py file")
        newconf = 1
    else:
        f1 = os.stat("%s/pyCGNSconfig.py" % (gfile))
        if (os.path.exists("%s/%s" % (pfile, userconfigfile))):
            f2 = os.stat("%s/%s" % (pfile, userconfigfile))
        else:
            f2 = os.stat("./%s" % userconfigfile)
        if (f1.st_mtime < f2.st_mtime):
            newconf = 1
            log("using modified %s file" % userconfigfile)
        else:
            newconf = 0
            log("using existing %s file" % userconfigfile)
    if newconf:
        sys.path = ['..'] + ['.'] + sys.path
        import setup_userConfig as UCFG  # USER CONFIG
        for ck in dir(UCFG):
            if (ck[0] != '_'): config_default[ck] = UCFG.__dict__[ck]
        if (not os.path.exists('%s' % gfile)):
            os.makedirs('%s' % gfile)
        f = open("%s/pyCGNSconfig.py" % (gfile), 'w+')
        f.writelines(config_default['file_pattern'] % config_default)
        f.close()


# --------------------------------------------------------------------
def frompath_HDF5():
    h5p = which("h5dump")
    if h5p is not None:
        h5root = '/'.join(h5p.split('/')[:-2])
    else:
        h5root = '/usr/local'
    return h5root

def guess_path_python():
    return os.path.dirname(sys.executable)

# --------------------------------------------------------------------
def find_HDF5(pincs, plibs, libs):
    notfound = 1
    extraargs = []
    vers = ''
    h5root = frompath_HDF5()
    pincs += [h5root, '%s/include' % h5root]
    if sys.platform == 'win32':
        pth = guess_path_python()
        pincs += [h5root, '{}\\Library\\include'.format(pth)]
    plibs += [h5root, '%s/lib64' % h5root]
    plibs += [h5root, '%s/lib' % h5root]
    if sys.platform == 'win32':
        pth = guess_path_python()
        plibs += [h5root, '{}\\Library\\lib'.format(pth)]
    pincs = unique_but_keep_order(pincs)
    plibs = unique_but_keep_order(plibs)
    for pth in plibs:
        if ((os.path.exists(pth + '/libhdf5.a'))
            or (os.path.exists(pth + '/libhdf5.so'))
            or (os.path.exists(pth + '/libhdf5.lib'))
            or (os.path.exists(pth + '/libhdf5.sl'))):
            notfound = 0
            plibs = [pth]
            break
    if notfound:
        log("***** FATAL: libhdf5 not found, please check paths:")
        for ppl in plibs:
            print(pfx, ppl)
    notfound = 1
    for pth in pincs:
        if (os.path.exists(pth + '/hdf5.h')): notfound = 0
    if notfound:
        log("***** FATAL: hdf5.h not found, please check paths")
        for ppi in pincs:
            print(pfx, ppi)
        return None

    ifh = 'HDF5 library version: unknown'
    notfound = 1
    for pth in pincs:
        if (os.path.exists(pth + '/H5public.h')):
            fh = open(pth + '/H5public.h', 'r')
            fl = fh.readlines()
            fh.close()
            found = 0
            for ifh in fl:
                if (ifh[:21] == "#define H5_VERS_INFO "):
                    vers = ifh.split('"')[1].split()[-1]
                    found = 1
            if found:
                pincs = [pth]
                notfound = 0
                break
    if notfound:
        log("***** FATAL: cannot find hdf5 version, please check paths")
        for ppi in pincs:
            print(pfx, pincs)
        return None

    h64 = 0
    hup = 1
    hst = 1
    if (os.path.exists(pth + '/H5pubconf.h')):
        hup = 1
        hst = 1
        hfn = pth + '/H5pubconf.h'
    if (os.path.exists(pth + '/h5pubconf.h')):
        hup = 0
        hst = 1
        hfn = pth + '/h5pubconf.h'
    if (os.path.exists(pth + '/H5pubconf-64.h')):
        h64 = 1
        hup = 1
        hfn = pth + '/H5pubconf-64.h'
    if (os.path.exists(pth + '/h5pubconf-64.h')):
        h64 = 1
        hup = 0
        hfn = pth + '/h5pubconf-64.h'
    has_parallel = False
    if (os.path.exists(hfn)):
        fh = open(hfn, 'r')
        fl = fh.readlines()
        fh.close()
        found = 0
        for ifh in fl:
            if (ifh[:26] == "#define H5_HAVE_PARALLEL 1"):
                has_parallel = True
    return (vers, pincs, plibs, libs, extraargs, hst, h64, hup, has_parallel)

# --------------------------------------------------------------------
def find_numpy(pincs, plibs, libs):
    try:
        import numpy
    except ImportError:
        log("**** FATAL cannot import numpy")
        sys.exit(0)
    apivers = ''
    vers = numpy.version.version
    extraargs = []
    pdir = os.path.normpath(sys.prefix)
    xdir = os.path.normpath(sys.exec_prefix)
    if sys.platform == 'win32':
        pth = guess_path_python()
        pincs += ['{}\\Library\\include'.format(pth)]
    pincs += ['%s/lib/python%s/site-packages/numpy/core/include' \
              % (xdir, sys.version[:3])]
    pincs += ['%s/lib/python%s/site-packages/numpy/core/include' \
              % (pdir, sys.version[:3])]
    pincs += [numpy.get_include()]
    notfound = 1
    pincs = unique_but_keep_order(pincs)
    plibs = unique_but_keep_order(plibs)
    for pth in pincs:
        if (os.path.exists(pth + '/numpy/ndarrayobject.h')):
            fh = open(pth + '/numpy/ndarrayobject.h', 'r')
            fl = fh.readlines()
            fh.close()
            found = 0
            for ifh in fl:
                if (ifh[:20] == "#define NPY_VERSION "):
                    apivers = ifh.split()[-1]
                    found = 1
            if found:
                pincs = [pth]
                notfound = 0
                break
        if (os.path.exists(pth + '/numpy/_numpyconfig.h')):
            fh = open(pth + '/numpy/_numpyconfig.h', 'r')
            fl = fh.readlines()
            fh.close()
            found = 0
            for ifh in fl:
                if (ifh[:24] == "#define NPY_ABI_VERSION "):
                    apivers = ifh.split()[-1]
                    found = 1
            if found:
                pincs = [pth]
                notfound = 0
                break
    if notfound:
        log("***** FATAL: numpy headers not found, please check your paths")
        log(pincs)
        return None

    return (vers, apivers, pincs, plibs, libs, extraargs)


# --------------------------------------------------------------------
def touch(filename):
    now = time.time()
    os.utime(filename, (now, now))

# --- last line
