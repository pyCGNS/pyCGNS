#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.VAL.grammars.CGNS_VAL_USER_DEFAULT as CGV
import CGNS.VAL.parse.messages as CGM
import CGNS.PAT.cgnserrors as CGE
import CGNS.VAL.parse.findgrammar

import sys
import os


def listuserkeys(trace):
    s = ""
    if not trace:
        print("### Use -v (verbose) option to check if the found paths")
        print("### are into your PYTHONPATH. If not, grammar would not be")
        print("### used by CGNS.VAL even if detected...")
    dk = CGNS.VAL.parse.findgrammar.findAllUserGrammars(trace)
    for key in dk:
        s += "%-16s: %s\n" % (key, dk[key])
    return s


def getParser(trace, user):
    if user is not None:
        mod = CGNS.VAL.parse.findgrammar.importUserGrammars(user, verbose=trace)
        if mod is None:
            if trace:
                print("### Using grammar [DEFAULT]")
            parser = CGV.CGNS_VAL_USER_Checks(None)
        else:
            if trace:
                print("### Using grammar [%s]" % user)
            if trace:
                print("### Found in [%s]" % os.path.dirname(mod.__file__))
            parser = mod.CGNS_VAL_USER_Checks(None)
    else:
        parser = CGV.CGNS_VAL_USER_Checks(None)
    return parser


def listdiags(trace, userlist):
    ld = []
    for user in userlist:
        parser = getParser(trace, user)
        mlist = parser.listDiagnostics()
        ld += list(mlist)
    ld.sort()
    s = ""
    for d in ld:
        s += "%s\n" % (str(mlist[d]))
    return s


def run(T, trace, userlist, stop=False, warnings=[], failures=[]):
    diag = CGM.DiagnosticLog()
    for user in userlist:
        parser = getParser(trace, user)
        # parser lookup and init are required prior to initialize message tables
        for w in warnings:
            diag.forceAsWarning(w)
        for w in failures:
            diag.forceAsFailure(w)
        try:
            parser.checkTree(T, trace, stop=stop)
        except (CGE.cgnsException,) as v:
            pass
        diag.merge(parser.log)
    return diag


def compliant(
    T,
    trace=False,
    userlist=["DEFAULT"],
    paths=[""],
    stop=False,
    warnings=[],
    failures=[],
):
    ipath = "%s/lib/python%s.%s/site-packages/CGNS/VAL/grammars" % (
        sys.prefix,
        sys.version_info[0],
        sys.version_info[1],
    )
    sys.path.append(ipath)
    for pp in paths:
        sys.path.append(pp)
    diag = run(T, trace, userlist, stop=stop, warnings=warnings, failures=failures)
    ok = [True, []]
    for p in diag:
        for (s, sp) in diag.diagnosticsByPath(p):
            if diag.status(s) == CGM.CHECK_FAIL:
                ok[0] = False
                ok[1].append((p, diag.message(s)))
    return ok


def showDiag(diag, idlist, bypath=True):
    ok = True
    if bypath:
        for p in diag:
            if not diag.hasOnlyKey(p, idlist):
                print("\n%s\n%s" % ("-" * 75, p))
                for (s, sp) in diag.diagnosticsByPath(p):
                    if (diag.status(s) != CGM.CHECK_GOOD) and (
                        diag.key(s) not in idlist
                    ):
                        print(diag.message(s))
                        if diag.status(s) == CGM.CHECK_FAIL:
                            ok = False
        print("\n%s\n" % ("-" * 75))
    else:
        for m in diag.allMessageKeys():
            if m not in idlist:
                first = True
                ctxt = diag.noContextMessage(m)
                if ctxt is not None:
                    print("\n%s\n[%s] %s" % ("-" * 75, m, ctxt))
                else:
                    print("\n%s\n[%s]" % ("-" * 75, m))
                for (d, dp) in diag.diagnosticsByMessage(m):
                    if diag.status(d) != CGM.CHECK_GOOD:
                        if ctxt is None:
                            if not first:
                                skip = "\n"
                            else:
                                skip = ""
                            print("%s  %s\n  > %s" % (skip, dp, d.message))
                        else:
                            print("  %s" % (dp))
                        first = False
                        if diag.status(d) == CGM.CHECK_FAIL:
                            ok = False
        print("\n%s" % ("-" * 75))
    if ok:
        print("### CGNS/Python tree Compliant")
    else:
        print("### CGNS/Python tree *NOT* Compliant")
