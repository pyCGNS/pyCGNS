#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import shutil
import os
import sys

OPTIONSFILENAME='.cgnsnavoptions.py'

flag=0
opath=os.environ['HOME']
if (os.path.exists(opath+'/'+OPTIONSFILENAME)):
  dpath='/tmp/%s%d'%(OPTIONSFILENAME,os.getpid())
  os.mkdir(dpath)
  shutil.copyfile(opath+'/'+OPTIONSFILENAME,dpath+'/s7options.py')
  sprev=sys.path
  sys.path=[dpath]+sys.path
  try:
    import s7options
  except ImportError:
    pass
  shutil.rmtree(dpath)
  sys.path=sprev
  flag=1

class s7optionsStorage:
  def __init__(self):
    # --------------------------------------------------------------------
    self.iconKeys=[
     'node-sids-opened','node-sids-closed','save','save-done',
     'pattern','pattern-open','pattern-close','pattern-reload',
     'operate-save','operate-add','operate-execute','operate-probe',
     'data-array-large','node-sids-leaf','operate-edit','check-view',
     'link-node','link-error','link-ignore','select-save','select-add',
     'node-sids-modified','check-fwd','check-bwd','mark-node',
     'mandatory-sids-node','mandatory-profile-node',
     'optional-sids-node' ,'help-view','options-view',
     'subtree-sids-ok', 'subtree-sids-warning','subtree-sids-failed',
     'pattern-save','tree-save','check-clear','check-save','check-all',
     'print-view','close-view','undo-last-modification','view-help',
     'flag-revert','flag-none','flag-all','operate-view','operate-list',
     'flag-bwd','flag-fwd','select-update',
     'level-in','level-out','tree-new','tree-add','tree-del','tree-load',
     'snapshot','pattern-view','link-view','query-and','query-or','query-not'
     ]
    # Next line is replaced each time you run distutils...
    # then s7globals_.py becomes... s7globals.py (good job boy !) 
    self.s7icondirectoryprefix='.' 
    self.iconDir=self.s7icondirectoryprefix+'/share/CGNS/NAV/icons'
    self.firedox='firefox %s/share/CGNS/NAV/doc/CGNSNAV.html &'%\
                  self.s7icondirectoryprefix
    self.profilePathSIDS=["%s/lib/python%d.%d/site-packages%s"%\
                          (sys.prefix,sys.version_info[0],sys.version_info[1],\
                           '/CGNS/PAT')]

    self.font={}
    self.iconStorage={}
    self.treeStorage=[]
    self.expandRecurse=0
    self.lateLoading=1    
    self.maxRecurse=30
    self.sidsRecurse=0
    self.singleView=0
    self.flyCheck=1
    self.maxDisplaySize=37
    self.minFileSizeNoWarning=1000000000
    self.noData=0
    self.forceFortranFlag=0
    self.transposeOnViewEdit=0        
    self.compactedValue=0
    self.showIndex=0
    self.showColumnTitle=1
    self.helpBallooons=1
    self.wControl=None
    self.generateCopyNames=1
    self.viewListFrame=None
    self.patternViewOpen=0
    self.defaultProfile='SIDS'
    self.printTag='CGNS.NAV:'
    self.followLinks=1
    self.saveLinks=1
    self.historyFile='.cgnsnavhistory.py'    
    self.showSIDS=0
    self.directoriesHistory=[]
    self.filesHistory=[]
    self.directoriesHistorySize=20
    self.operateListSplit=0
    self.profilePath=[]

    self.shiftstring='  '
    self.wminwidth=700
    self.wminheight=150    

    self.cgnspyFiles =['','.py','.pyc','.pyo']
    self.cgnsbinFiles=['.hdf','.hdf5','.cgns','.adf']

    self.cgnslibFiles=['.cgns','.adf']
    self.cgnssslFiles=['.hdf','.hdf5']

    self.SIDSkeyword='keyword'
    self.SIDSoptional='optional'
    self.SIDSmandatory='mandatory'
    self.USERoptional='user-optional'
    self.USERmandatory='user-mandatory'

    self.SIDSmodes=[self.SIDSoptional,self.SIDSmandatory,
                    self.USERoptional,self.USERmandatory]

    # --------------------------------------------------------------------
    # Fonts
    # family -- font 'family', e.g. Courier, Times, Helvetica
    # size   -- font size in points
    # weight -- font thickness: normal, bold
    # slant  -- font slant: roman, italic
    #
    # --------------------------------------------------
    def createFontDict(d,ff):
      import tkFont
      for k in ff:
        d[k]=tkFont.Font(family=ff[k][0],
                         weight=ff[k][2],
                         size=ff[k][1],
                         slant=ff[k][3])
      return d

    createFontDict(self.font,{
    'M' :['Helvetica',10,'normal','roman'], # All menus
    'E' :['Courier',  10,'normal','roman'], # Entries/Text
    'F' :['Courier',  10,'bold','roman'],   # Entries/Text bold
    'X' :['Courier',   8,'normal','roman'], # not used
    'H' :['Courier',  10,'bold',  'italic'],# All tables headers
    'L' :['Helvetica',10,'bold',  'roman'], # Various labels
    'B' :['Helvetica',10,'normal','italic'],# Ok/Save... buttons

    'Ta':['Helvetica',12,'normal','roman'], # Tree- SIDS node name
    'Tb':['Helvetica',12,'bold',  'roman'], # Tree- user node name
    'Tc':['Helvetica',10,'normal','roman'], # Tree- Node type
    'Td':['Courier',  10,'normal','roman'], # Tree- Data
    'Te':['Courier',  10,'bold',  'roman'], # Tree- Status (path)
    'Tf':['Courier',  10,'normal','roman'], # Tree- Data type
     })

    # --------------------------------------------------------------------
    # Colors
    self.color_S7='#d8d8d8'  # All widgets background (default gray)
    
    self.color_Ca='#c9ffb6'  # Control view - Line selection foreground
    self.color_Cb='Black'    # Control view - Line selection background

    self.color_Ta='SeaGreen' # Tree view - Node name font - SIDS name
    self.color_Tb='SeaGreen' # Tree view - Node name font - user name
    self.color_Tc='#c9ffb6'  # Tree view - End of line rectangle
    self.color_Tm='Red'      # Tree view - Mandatory node

    self.allowedoptionslist=[]
    self.allowedoptionslist+=['expandRecurse']
    self.allowedoptionslist+=['maxRecurse']
    self.allowedoptionslist+=['sidsRecurse']
    self.allowedoptionslist+=['lateLoading']    
    self.allowedoptionslist+=['singleView']
    self.allowedoptionslist+=['flyCheck']
    self.allowedoptionslist+=['maxDisplaySize']
    self.allowedoptionslist+=['showIndex']
    self.allowedoptionslist+=['showColumnTitle']
    self.allowedoptionslist+=['defaultProfile']
    self.allowedoptionslist+=['profilePath']
    self.allowedoptionslist+=['followLinks']
    self.allowedoptionslist+=['saveLinks']
    self.allowedoptionslist+=['noData']
    self.allowedoptionslist+=['forceFortranFlag']
    self.allowedoptionslist+=['transposeOnViewEdit']    
    self.allowedoptionslist+=['compactedValue']
    self.allowedoptionslist+=['showSIDS']        
    self.allowedoptionslist+=['helpBallooons']        

  def update(self):
    # --------------------------------------------------------------------
    # local options overwrite previous globals

    global flag
    if (flag):
      flag=0
      for v in self.allowedoptionslist:
        if (hasattr(s7options,v)):
          setattr(self,v,getattr(s7options,v))

    self.profilePath+=self.profilePathSIDS

    #else:
    #  pass
    #  # reload(s7options)

s7G=s7optionsStorage()
s7G.update()

# --------------------------------------------------------------------

