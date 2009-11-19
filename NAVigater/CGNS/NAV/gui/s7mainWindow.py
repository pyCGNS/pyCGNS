#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import Tkinter
Tkinter.wantobjects=0 
from Tkinter import *
import os

def run(filelist,recurse,ppath,verbose):
  wtop=Tk()
  wtop.withdraw()
  wtop.option_add('*font', 'Sony 14 bold')

  import s7globals
  G___=s7globals.s7G

  wtop.tk_setPalette(G___.color_S7)
  if (recurse): G___.expandRecurse=1

  if (ppath != ""): G___.profilePath=ppath.split(':')+G___.profilepath

  import s7viewControl

  G___.wControl=s7viewControl.wTopControl(wtop)
  for f in filelist:
    if (os.path.exists(f)): G___.wControl.loadFile(f)
    else: print "## pyS7: No such file [%s]"%f

  try:
    wtop.mainloop()
  except KeyboardInterrupt:
    pass

# --------------------------------------------------------------------
