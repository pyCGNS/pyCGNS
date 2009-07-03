# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 70 $ $Date: 2009-01-30 11:49:10 +0100 (Fri, 30 Jan 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.

print 'SETUP NAV'

from distutils.core import setup, Extension
import glob
import sys
import os
import re
sys.path.append('./S7')
from version import __vid__
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--prefix",dest="prefix")
try:
  (options, args) = parser.parse_args(sys.argv)
except optparse.OptionError: pass

icondirprefix=sys.prefix
try:
  if (options.prefix != None): icondirprefix=options.prefix
  fg=open("./S7/tk/s7globals_.py",'r')
  llg=fg.readlines()
  fg.close()
  gg=open("./S7/tk/s7globals.py",'w+')
  for lg in llg:
    if (lg[:31]=='    self.s7icondirectoryprefix='):
      gg.write('    self.s7icondirectoryprefix="%s"\n'%icondirprefix)
    else:
      gg.write(lg)
  gg.close()
except KeyError: pass

def writeDestFile(file,l):
  try:
    os.mkdir('build')
  except OSError: pass
  try:
    os.mkdir('build/doc')
  except OSError: pass
  f=open('build/doc/'+file,'w+')
  f.writelines(l)
  f.close()

def readFile(file):
  f=open('S7/doc/'+file,'r')
  ll=f.readlines()
  f.close()
  return ll

def readSourceFile(file,title=None,img=None):
  l=[]
  if (title and img):
    l=["""%s</b></td>
         <td align="center"><img src="%s" align="right" border=0 />
         </td><tr /></table>"""%(title,img)]
  if (title and not img):
    l=["""%s</b></td>
          <td align="center">
          </td><tr /></table>"""%(title)]
  l+=readFile(file)
  return l

def produceDoc(tag,title,img=None):
  ltop=readSourceFile('frametop.ht')
  lbot=readSourceFile('framebot.ht')
  lt=ltop+readSourceFile('%s.ht'%tag,title,img)+lbot
  writeDestFile('%s.html'%tag,lt)  

s7title='SIDS Surveyor and Sketcher with Several Services and Specializable Supervisor'

produceDoc('pyS7',s7title,'s7.png')
produceDoc('install','Installation')
produceDoc('whatsS7','What is S7?')
produceDoc('sevensteps','Seven Steps Tutorial')
produceDoc('ref','Reference')
produceDoc('cgns','CFD General Notation System')
produceDoc('about','About S7')
produceDoc('views','Surveyor and sketcher')
produceDoc('scop','Python/CGNS trees (SCOP)')
produceDoc('pycgns','pyCGNS module')
produceDoc('hdf5','HDF5 and ADF')
produceDoc('vcontrol','Control View')
produceDoc('vtree','Tree View')
produceDoc('vtable','Table View')
produceDoc('vpattern','Pattern View')
produceDoc('voperate','Operate View')
produceDoc('vselection','Selection View')
produceDoc('vlink','Link View')
produceDoc('none','Empty page')
  
# --------------------------------------------------------------------
# Clean target redefinition - force clean everything
from distutils.dir_util import remove_tree
from distutils.command.clean import clean as _clean

relist=['^.*~$','^#.*#$','^.*\.aux$','^.*\.pyc$','^.*\.bak$']
reclean=[]

for restring in relist:
  reclean.append(re.compile(restring))

def wselect(args,dirname,names):
  for n in names:
    for rev in reclean:
      if (rev.match(n)):
        #print "%s/%s"%(dirname,n)
        os.remove("%s/%s"%(dirname,n))
        break

class clean(_clean):
  def walkAndClean(self):
    os.path.walk(".",wselect,[])
  def run(self):
    if os.path.exists("./build"): remove_tree("./build")
    self.walkAndClean()



setup (
name         = "pyS7",
version      = __vid__,
description  = s7title,
author       = "marc.poinot@onera.fr",

packages     = ['S7','S7/tk','S7/supervisor'],

scripts      = ['S7/s7','S7/s7supervisor'],
data_files   = [('share/CGNS/S7/icons',glob.glob('S7/tk/icons/*')),
                ('share/CGNS/S7/doc',glob.glob('build/doc/*')),
                ('share/CGNS/S7/doc',glob.glob('S7/doc/*.png')),
                ('share/CGNS/S7/doc/img',glob.glob('S7/doc/img/*.png')),
                ('share/CGNS/S7/doc/img',glob.glob('S7/tk/icons/*')),      
                ],

cmdclass={'clean': clean}
)
 
