#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import sys
import os
import os.path
import fnmatch
import importlib
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
    try:
        grammars_spec = importlib.util.find_spec("grammars")
    except (ImportError, AttributeError, TypeError, ValueError) as ex:
        raise Exception("Error while finding module specification for grammars")
    if not grammars_spec:
        raise Exception("Could not find grammars module")
    mod = importlib.util.module_from_spec(grammars_spec)
    grammars_spec.loader.exec_module(mod)
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
        tp_spec = importlib.util.find_spec(modname)
    except (ImportError, AttributeError, TypeError, ValueError) as ex:
        raise Exception("Error while finding module specification for %s" % modname)
    if not tp_spec:
        if verbose:
            print("### Error: grammar [%s] not found" % key)
        if recurse:
            dk = findOneUserGrammar(key)
            if key in dk:
                sys.path.append(dk[key])
                if verbose:
                    print("### Warning: not in search path [%s]" % dk[key])
                try:
                    tp_spec = importlib.util.find_spec(modname)
                except (ImportError, AttributeError, TypeError, ValueError) as ex:
                    raise Exception(
                        "Error while finding module specification for %s" % modname
                    )
                if not tp_spec:
                    return None
            else:
                return None
        else:
            return None
    mod = importlib.util.module_from_spec(tp_spec)
    tp_spec.loader.exec_module(mod)
    # sys.modules[modname] = mod
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
