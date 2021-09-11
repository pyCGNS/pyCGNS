#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import sys
import os
import os.path
import fnmatch
import imp
import CGNS.VAL.grammars.etablesids as STB

PROFILENAME = "grammars"


#  -------------------------------------------------------------------------
def readProfile():
    try:
        hdir = os.environ["HOME"]
    except:
        return {}
    pdir = "%s%s.CGNS.NAV" % (hdir, os.path.sep)
    if not os.path.exists(pdir):
        return {}
    sys.path.append(pdir)
    fp, pth, des = imp.find_module("grammars")
    try:
        mod = imp.load_module("grammars", fp, pth, des)
    finally:
        if fp:
            fp.close()
    return mod.Grammars


#  -------------------------------------------------------------------------
def findAllUserGrammars(verbose=False):
    kdict = {}
    for pth in [p for p in sys.path if p != ""]:
        if verbose:
            print("### scanning", pth)
        try:
            for pthroot, dirs, files in os.walk(pth):
                for fn in files:
                    if fnmatch.fnmatch(fn, "CGNS_VAL_USER_*.py") or fnmatch.fnmatch(
                        fn, "CGNS_VAL_USER_*.so"
                    ):
                        gkey = fn[14:-3]
                        if gkey in kdict:
                            if verbose:
                                print(
                                    "### * found grammar:",
                                )
                                print(gkey, "already found, ignore this one")
                        else:
                            if verbose:
                                print("### * found grammar:", gkey)
                            kdict[fn[14:-3]] = pthroot
                        if verbose:
                            print("### * found in :", pthroot)
                            if pthroot not in sys.path:
                                print("### * previous path is NOT in PYTHONPATH")
                            else:
                                print("### * previous path already is in PYTHONPATH")

        except OSError:
            pass
    return kdict


#  -------------------------------------------------------------------------
def findOneUserGrammar(tag, verbose=False):
    kdict = {}
    found = False
    for pth in sys.path:
        if verbose:
            print("### scanning", pth)
        try:
            for pthroot, dirs, files in os.walk(pth):
                for fn in files:
                    if fnmatch.fnmatch(fn, "CGNS_VAL_USER_%s.py" % tag):
                        kdict[fn[14:-3]] = pthroot
                        found = True
                        break
                if found:
                    break
        except OSError:
            pass
        if found:
            break
    return kdict


#  -------------------------------------------------------------------------
def importUserGrammars(key, recurse=False, verbose=False):
    mod = None
    modname = "CGNS_VAL_USER_%s" % key
    ipath = os.path.split(STB.__file__)[0]
    sys.path.append(ipath)

    if verbose:
        print("### Looking for grammar [%s]" % key)
    try:
        tp = imp.find_module(modname)
    except ImportError:
        if verbose:
            print("### Error: grammar [%s] not found" % key)
        if recurse:
            dk = findOneUserGrammar(key)
            if key in dk:
                sys.path.append(dk[key])
                if verbose:
                    print("### Warning: not in search path [%s]" % dk[key])
                try:
                    tp = imp.find_module(modname)
                except ImportError:
                    return None
            else:
                return None
        else:
            return None
    try:
        fp = tp[0]
        if tp[2][2] != imp.C_EXTENSION:
            mod = imp.load_module(modname, *tp)
        else:
            # print '### CGNS.VAL [info]: Module info',tp
            mod = imp.load_dynamic(modname, tp[1], tp[0])
    except:
        pass
    finally:
        if fp:
            fp.close()
    return mod


#  -------------------------------------------------------------------------
def locateGrammars():
    glist = []
    for k in ["SIDS", "elsA"]:
        mod = importUserGrammars(k)
        if mod is not None:
            glist.append((k, mod.__file__))
    return glist


# ---
