#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
#
import re

class DataFile:
  def __init__(self):
    self.__doc=None

  def readFileAndParse(self,filename):
    import libxml2
    self.__doc = libxml2.parseFile(filename)
    return self.__doc

  def getRoot(self):
    return self.__doc.children

  # node -> name, type, content, next, last, prev, doc, properties

class Schema2:
    def __init__(self,fileschema):
      schema_name=fileschema
      input = open(schema_name, 'r')
      schema = input.read()
      input.close()

      rngp = libxml2.relaxNGNewMemParserCtxt(schema, len(schema))
      rngs = rngp.relaxNGParse()
        
    def parse(self):
        for n in schemaDom.childNodes:
           if (n.attributes):
             nameN=n.attributes.getNamedItem("name")
             if (nameN):
               name=nameN.nodeValue
               print n.nodeName, name
           self.parse(n)

class Schema:
    tEnumerate=0
    tType=1
    
    def __init__(self,domschema):
      self.dom=domschema
      self.grammar={}
      
    def parse(self,node=None):
      if (not node):
        node=self.dom
      #print node.nodeName
      for n in node.childNodes:
        #print n
        parsed=0
        nn=n.nodeName
        if (nn=="include"):
           sn=self.readInclude(n.attributes.getNamedItem("href").nodeValue)
           self.parse(sn)
           parsed=1
        if (nn=="define"):
           typeKey=n.attributes.getNamedItem("name").nodeValue
           firstSon=n.childNodes[1]
           if (firstSon.nodeName=="choice"):
             # got an enumerate
             self.parseChoice(typeKey,firstSon)
             parsed=1
           if (firstSon.nodeName=="element"):
             # got a type
             self.parseType(typeKey,firstSon)
             parsed=1
        if (not parsed):
           self.parse(n)
         
    def parseChoice(self,name,node):
      l=[]
      self.grammar[name]=[self.tEnumerate, l ]
      #print name, self.grammar[name]
      for n in node.childNodes:
        nn=n.nodeName
        if (nn=="value"):
          l.append(str(n.childNodes[0].nodeValue))
        
    def parseType(self,name,node):
      l=[]
      self.grammar[name]=[self.tType, l ]

    def readInclude(self,url):
      #print url
      readerI = PyExpat.Reader()
      return readerI.fromUri(url)

        
