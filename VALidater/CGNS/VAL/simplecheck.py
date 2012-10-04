#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import CGNS.VAL.grammars.SIDS
import CGNS.VAL.grammars.USER
import CGNS.VAL.parse.messages as CGM

def run(T,trace,user):
    if (user):
        parser=CGNS.VAL.grammars.USER.elsAbase(None)        
    else:
        parser=CGNS.VAL.grammars.SIDS.SIDSbase(None)
    parser.checkTree(T,trace)
    return parser

def showDiag(diag,idlist):
    sz=len(diag.log)
    ct=1
    ok=True
    for p in diag.log:
        if (not diag.log.hasOnlyKey(p,idlist)):
          print '\n%s\n%s'%('-'*75,p)
          for s in diag.log.diagnostics(p):
              if ((diag.log.status(s)!=CGM.CHECK_GOOD)
                  and (diag.log.key(s) not in idlist)):
                  print diag.log.message(s)
                  if (diag.log.status(s)==CGM.CHECK_FAIL): ok=False
              ct+=1
    print '\n%s\n'%('-'*75)
    if (ok): print '\n### CGNS/Python tree Compliant'
    else:    print '\n### CGNS/Python tree *NOT* Compliant'
    
