#
# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyCCCCC - $Id: rt.py 22 2005-02-02 09:57:08Z  $
#
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
#
javadir="/home/poinot/W/Xpy/RelaxNG/jar"
cpstr  ="%s/trang.jar:"%javadir
cpstr +="%s/jing.jar:"  %javadir
cpstr +="%s/crimson.jar"%javadir

R0="java -cp %s"%(cpstr)
Rdriver="com.thaiopensource.relaxng.translate.Driver"
Rcheck="com.thaiopensource.relaxng.util.Driver"
Rtranslation="%s %s"%(R0,Rdriver)+" %s %s"
Rvalidation="%s %s"%(R0,Rcheck)+" %s %s"

import os

def translate(inf,outf):
  Rx=Rtranslation%(inf,outf)
  print Rx
  os.system(Rx)

def validate(inf,outf):
  Rx=Rvalidation%(inf,outf)
  print Rx
  os.system(Rx)
