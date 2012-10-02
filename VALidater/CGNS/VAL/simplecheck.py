#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import CGNS.VAL.grammars.SIDS  as CGS
import CGNS.VAL.parse.messages as CGM

def run(T,trace):
    parser=CGS.SIDSbase()
    parser.checkTree(T,trace)
    return parser

def pline(ct,sz,node):
    print '\n### [%.6d/%.6d] %s'%(ct,sz,node)
    
def showDiag(diag):
    sz=len(diag.log)
    ct=1
    ok=True
    for p in diag.log:
        for s in diag.log.diagnostics(p):
            if (s[0]!=CGM.CHECK_GOOD):
                print diag.log.pline(p,s)
                ok=False
            ct+=1
    if (ok): print '\n### CGNS/Python tree Compliant'
    else:    print '\n### CGNS/Python tree *NOT* Compliant'
    
