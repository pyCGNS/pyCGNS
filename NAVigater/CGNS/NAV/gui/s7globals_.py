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
     'flag-bwd','flag-fwd','select-update','vtk',
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

    if (self.profilePathSIDS[0] not in self.profilePath):
      self.profilePath+=self.profilePathSIDS

    #else:
    #  pass
    #  # reload(s7options)

  colors={\
     #  Greys
     'cold_grey': (0.5000, 0.5400, 0.5300),
     'dim_grey': (0.4118, 0.4118, 0.4118),
     'grey': (0.7529, 0.7529, 0.7529),
     'light_grey': (0.8275, 0.8275, 0.8275),
     'slate_grey': (0.4392, 0.5020, 0.5647),
     'slate_grey_dark': (0.1843, 0.3098, 0.3098),
     'slate_grey_light': (0.4667, 0.5333, 0.6000),
     'warm_grey': (0.5000, 0.5000, 0.4100),
     #  Blacks
     'black': (0.0000, 0.0000, 0.0000),
     'ivory_black': (0.1600, 0.1400, 0.1300),
     'lamp_black': (0.1800, 0.2800, 0.2300),
     #  Reds
     'alizarin_crimson': (0.8900, 0.1500, 0.2100),
     'brick': (0.6100, 0.4000, 0.1200),
     'cadmium_red_deep': (0.8900, 0.0900, 0.0500),
     'coral': (1.0000, 0.4980, 0.3137),
     'coral_light': (0.9412, 0.5020, 0.5020),
     'deep_pink': (1.0000, 0.0784, 0.5765),
     'english_red': (0.8300, 0.2400, 0.1000),
     'firebrick': (0.6980, 0.1333, 0.1333),
     'geranium_lake': (0.8900, 0.0700, 0.1900),
     'hot_pink': (1.0000, 0.4118, 0.7059),
     'indian_red': (0.6900, 0.0900, 0.1200),
     'light_salmon': (1.0000, 0.6275, 0.4784),
     'madder_lake_deep': (0.8900, 0.1800, 0.1900),
     'maroon': (0.6902, 0.1882, 0.3765),
     'pink': (1.0000, 0.7529, 0.7961),
     'pink_light': (1.0000, 0.7137, 0.7569),
     'raspberry': (0.5300, 0.1500, 0.3400),
     'red': (1.0000, 0.0000, 0.0000),
     'rose_madder': (0.8900, 0.2100, 0.2200),
     'salmon': (0.9804, 0.5020, 0.4471),
     'tomato': (1.0000, 0.3882, 0.2784),
     'venetian_red': (0.8300, 0.1000, 0.1200),
     #  Browns
     'beige': (0.6400, 0.5800, 0.5000),
     'brown': (0.5000, 0.1647, 0.1647),
     'brown_madder': (0.8600, 0.1600, 0.1600),
     'brown_ochre': (0.5300, 0.2600, 0.1200),
     'burlywood': (0.8706, 0.7216, 0.5294),
     'burnt_sienna': (0.5400, 0.2100, 0.0600),
     'burnt_umber': (0.5400, 0.2000, 0.1400),
     'chocolate': (0.8235, 0.4118, 0.1176),
     'deep_ochre': (0.4500, 0.2400, 0.1000),
     'flesh': (1.0000, 0.4900, 0.2500),
     'flesh_ochre': (1.0000, 0.3400, 0.1300),
     'gold_ochre': (0.7800, 0.4700, 0.1500),
     'greenish_umber': (1.0000, 0.2400, 0.0500),
     'khaki': (0.9412, 0.9020, 0.5490),
     'khaki_dark': (0.7412, 0.7176, 0.4196),
     'light_beige': (0.9608, 0.9608, 0.8627),
     'peru': (0.8039, 0.5216, 0.2471),
     'rosy_brown': (0.7373, 0.5608, 0.5608),
     'raw_sienna': (0.7800, 0.3800, 0.0800),
     'raw_umber': (0.4500, 0.2900, 0.0700),
     'sepia': (0.3700, 0.1500, 0.0700),
     'sienna': (0.6275, 0.3216, 0.1765),
     'saddle_brown': (0.5451, 0.2706, 0.0745),
     'sandy_brown': (0.9569, 0.6431, 0.3765),
     'tan': (0.8235, 0.7059, 0.5490),
     'van_dyke_brown': (0.3700, 0.1500, 0.0200),
     #  Oranges
     'cadmium_orange': (1.0000, 0.3800, 0.0100),
     'cadmium_red_light': (1.0000, 0.0100, 0.0500),
     'carrot': (0.9300, 0.5700, 0.1300),
     'dark_orange': (1.0000, 0.5490, 0.0000),
     'mars_orange': (0.5900, 0.2700, 0.0800),
     'mars_yellow': (0.8900, 0.4400, 0.1000),
     'orange': (1.0000, 0.5000, 0.0000),
     'orange_red': (1.0000, 0.2706, 0.0000),
     'yellow_ochre': (0.8900, 0.5100, 0.0900),
     #  Yellows
     'aureoline_yellow': (1.0000, 0.6600, 0.1400),
     'banana': (0.8900, 0.8100, 0.3400),
     'cadmium_lemon': (1.0000, 0.8900, 0.0100),
     'cadmium_yellow': (1.0000, 0.6000, 0.0700),
     'cadmium_yellow_light': (1.0000, 0.6900, 0.0600),
     'gold': (1.0000, 0.8431, 0.0000),
     'goldenrod': (0.8549, 0.6471, 0.1255),
     'goldenrod_dark': (0.7216, 0.5255, 0.0431),
     'goldenrod_light': (0.9804, 0.9804, 0.8235),
     'goldenrod_pale': (0.9333, 0.9098, 0.6667),
     'light_goldenrod': (0.9333, 0.8667, 0.5098),
     'melon': (0.8900, 0.6600, 0.4100),
     'naples_yellow_deep': (1.0000, 0.6600, 0.0700),
     'yellow': (1.0000, 1.0000, 0.0000),
     'yellow_light': (1.0000, 1.0000, 0.8784),
     #  Greens
     'chartreuse': (0.4980, 1.0000, 0.0000),
     'chrome_oxide_green': (0.4000, 0.5000, 0.0800),
     'cinnabar_green': (0.3800, 0.7000, 0.1600),
     'cobalt_green': (0.2400, 0.5700, 0.2500),
     'emerald_green': (0.0000, 0.7900, 0.3400),
     'forest_green': (0.1333, 0.5451, 0.1333),
     'green': (0.0000, 1.0000, 0.0000),
     'green_dark': (0.0000, 0.3922, 0.0000),
     'green_pale': (0.5961, 0.9843, 0.5961),
     'green_yellow': (0.6784, 1.0000, 0.1843),
     'lawn_green': (0.4863, 0.9882, 0.0000),
     'lime_green': (0.1961, 0.8039, 0.1961),
     'mint': (0.7400, 0.9900, 0.7900),
     'olive': (0.2300, 0.3700, 0.1700),
     'olive_drab': (0.4196, 0.5569, 0.1373),
     'olive_green_dark': (0.3333, 0.4196, 0.1843),
     'permanent_green': (0.0400, 0.7900, 0.1700),
     'sap_green': (0.1900, 0.5000, 0.0800),
     'sea_green': (0.1804, 0.5451, 0.3412),
     'sea_green_dark': (0.5608, 0.7373, 0.5608),
     'sea_green_medium': (0.2353, 0.7020, 0.4431),
     'sea_green_light': (0.1255, 0.6980, 0.6667),
     'spring_green': (0.0000, 1.0000, 0.4980),
     'spring_green_medium': (0.0000, 0.9804, 0.6039),
     'terre_verte': (0.2200, 0.3700, 0.0600),
     'viridian_light': (0.4300, 1.0000, 0.4400),
     'yellow_green': (0.6039, 0.8039, 0.1961),
     #  Cyans
     'aquamarine': (0.4980, 1.0000, 0.8314),
     'aquamarine_medium': (0.4000, 0.8039, 0.6667),
     'cyan': (0.0000, 1.0000, 1.0000),
     'cyan_white': (0.8784, 1.0000, 1.0000),
     'turquoise': (0.2510, 0.8784, 0.8157),
     'turquoise_dark': (0.0000, 0.8078, 0.8196),
     'turquoise_medium': (0.2824, 0.8196, 0.8000),
     'turquoise_pale': (0.6863, 0.9333, 0.9333),
     #  Blues
     'alice_blue': (0.9412, 0.9725, 1.0000),
     'blue': (0.0000, 0.0000, 1.0000),
     'blue_light': (0.6784, 0.8471, 0.9020),
     'blue_medium': (0.0000, 0.0000, 0.8039),
     'cadet': (0.3725, 0.6196, 0.6275),
     'cobalt': (0.2400, 0.3500, 0.6700),
     'cornflower': (0.3922, 0.5843, 0.9294),
     'cerulean': (0.0200, 0.7200, 0.8000),
     'dodger_blue': (0.1176, 0.5647, 1.0000),
     'indigo': (0.0300, 0.1800, 0.3300),
     'manganese_blue': (0.0100, 0.6600, 0.6200),
     'midnight_blue': (0.0980, 0.0980, 0.4392),
     'navy': (0.0000, 0.0000, 0.5020),
     'peacock': (0.2000, 0.6300, 0.7900),
     'powder_blue': (0.6902, 0.8784, 0.9020),
     'royal_blue': (0.2549, 0.4118, 0.8824),
     'slate_blue': (0.4157, 0.3529, 0.8039),
     'slate_blue_dark': (0.2824, 0.2392, 0.5451),
     'slate_blue_light': (0.5176, 0.4392, 1.0000),
     'slate_blue_medium': (0.4824, 0.4078, 0.9333),
     'sky_blue': (0.5294, 0.8078, 0.9216),
     'sky_blue_deep': (0.0000, 0.7490, 1.0000),
     'sky_blue_light': (0.5294, 0.8078, 0.9804),
     'steel_blue': (0.2745, 0.5098, 0.7059),
     'steel_blue_light': (0.6902, 0.7686, 0.8706),
     'turquoise_blue': (0.0000, 0.7800, 0.5500),
     'ultramarine': (0.0700, 0.0400, 0.5600),
     #  Magentas
     'blue_violet': (0.5412, 0.1686, 0.8863),
     'cobalt_violet_deep': (0.5700, 0.1300, 0.6200),
     'magenta': (1.0000, 0.0000, 1.0000),
     'orchid': (0.8549, 0.4392, 0.8392),
     'orchid_dark': (0.6000, 0.1961, 0.8000),
     'orchid_medium': (0.7294, 0.3333, 0.8275),
     'permanent_red_violet': (0.8600, 0.1500, 0.2700),
     'plum': (0.8667, 0.6275, 0.8667),
     'purple': (0.6275, 0.1255, 0.9412),
     'purple_medium': (0.5765, 0.4392, 0.8588),
     'ultramarine_violet': (0.3600, 0.1400, 0.4300),
     'violet': (0.5600, 0.3700, 0.6000),
     'violet_dark': (0.5804, 0.0000, 0.8275),
     'violet_red': (0.8157, 0.1255, 0.5647),
     'violet_red_medium': (0.7804, 0.0824, 0.5216),
     'violet_red_pale': (0.8588, 0.4392, 0.5765),
       }

s7G=s7optionsStorage()
s7G.update()

# --------------------------------------------------------------------

