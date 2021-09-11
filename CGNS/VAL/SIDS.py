#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
import CGNS.VAL.simplecheck


def check(
    t, trace=False, grammar=None, paths=[""], stop=False, warnings=[], failures=[]
):
    if grammar is None:
        grammar = ["SIDS"]
    chk = CGNS.VAL.simplecheck.compliant(
        t,
        userlist=grammar,
        paths=paths,
        stop=stop,
        warnings=warnings,
        failures=failures,
    )
    if trace:
        for d in chk[1]:
            print("# %s >>> %s" % (d[1], d[0]))
    return chk[0]
