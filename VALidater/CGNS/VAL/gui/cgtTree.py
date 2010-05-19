# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyCCCCC - $Id: cgtTree.py 39 2005-10-19 13:37:18Z  $
#
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import Tree
import cgtParseSchema
from   Tkinter        import *
import CGNS
import re
import Icons.base as baseIcon
import string

#
import xml.dom.minidom as XmlDom
#
cgtTreeDefaultFont=('Courier',10)
cgtTreeDefaultFontAttribute=('Courier',8)
cgtTreeDefaultFontDialog=('Arial',10)
cgtTreeDefaultFontTitle=('Courier',10,'bold')
cgtTreeDefaultFontMenu=('Arial',10,'italic')
cgtTreeDefaultFontComment=('Arial',8,'italic')
#
CGNSNodeNameAttribute='Name'
#
CGNSNameList=CGNS.Names.values()+[
    'GridCoordinates',
    'ZoneBC',
    'ZoneGridConnectivity',
    'GlobalConvergenceHistory',
    'Transform',
    'PointRange',
    'PointRangeDonor',
    'ZoneType'
    ]
#
global GlobalSchemaGrammar
GlobalSchemaGrammar=None
#
class CGNSNode(Tree.Node):

    def __init__(self, *args, **kw_args):

      # call superclass
      apply(Tree.Node.__init__, (self,)+args, kw_args)
      self.shut_icon=PhotoImage(data=baseIcon.isOff)
      self.open_icon=PhotoImage(data=baseIcon.isOn)
      self.comment_icon=PhotoImage(data=baseIcon.comment)
      self.attribute_icon=PhotoImage(data=baseIcon.attribute)

      # --- mouse No1 button
      if self.indic: self.widget.tag_bind(self.indic,  '<1>', self.expandNode)
      self.widget.tag_bind(self.symbol, '<1>', self.expandNode)
      self.widget.tag_bind(self.label,  '<1>', self.expandNode)

      # --- mouse No2 button
      self.widget.tag_bind(self.symbol, '<2>', self.popup_menu1)
      self.widget.tag_bind(self.label,  '<2>', self.popup_menu1)

      # --- mouse No2 button + CTRL
      self.widget.tag_bind(self.symbol, '<Control-2>', self.markNode)
      self.widget.tag_bind(self.label,  '<Control-2>', self.markNode)

      # --- mouse No3 button
      self.widget.tag_bind(self.symbol, '<3>', self.popup_menu2)
      self.widget.tag_bind(self.label,  '<3>', self.popup_menu2)

      self.re_empty=re.compile(r"^[\n\t ]*$")

    def operateExtra(self,x,y):
        """Override Base method: Set current text label"""
        pass
        #if (len(self.id) > 4):
            #(dx,dy)=self.widget.coords(self.label)
            #self.box=self.widget.create_rectangle(x,y-8,x+100,y+8)

    # pop up menus on left/right click
    def popup_menu1(self, event):
      # empty file
      if (self.id[0] == None): return
      n=self.id[0]
      f=cgtTreeDefaultFontMenu
      menu=Menu(self.widget, tearoff=0)
      if (n.nodeType == XmlDom.Node.TEXT_NODE):
        s="Data menu"
      else:
        s="%s [%s]"%(val(n,CGNSNodeNameAttribute),n.nodeName)
      menu.add_command(label=s,font=cgtTreeDefaultFont)
      menu.add('separator')
      menu.add_command(label="Edit",font=f,command=self.editNode)
      menu.add('separator')
      menu.add_command(label="Copy",
                       font=f,
                       command=self.copyNode)
      menu.add_command(label="Paste as brother",
                       font=f,
                       command=self.pasteAsBrotherNode)
      menu.add_command(label="Paste as child",
                       font=f,
                       command=self.pasteAsChildNode)
      menu.add_command(label="Delete",
                       font=f,
                       command=self.deleteNode)
      menu.add('separator')
      menu.add_command(label="Link",font=f,command=self.linkNode)
      menu.add('separator')
      menu.add_command(label="Export",font=f,command=self.exportNode)
      menu.add_command(label="Import",font=f,command=self.importNode)      
      menu.tk_popup(event.x_root, event.y_root)

    def popup_menu2(self, event):
      # empty file
      if (self.id[0] == None): return
      n=self.id[0]
      if (n.nodeType == XmlDom.Node.TEXT_NODE): return
      if (n.nodeType == XmlDom.Node.COMMENT_NODE):
        #showc=self.widget.top.showcomment.get()
        menu=Menu(self.widget, tearoff=0)
        if (len(n.data) > 80):
          s=n.data[:80]
        else:
          s=n.data
        menu.add_command(label=s,font=cgtTreeDefaultFont)
        menu.tk_popup(event.x_root, event.y_root)
      else:
        dft=cgtTreeDefaultFont
        shwa=1 # force display for this menu
        menu=Menu(self.widget, tearoff=0)
        s="%s [%s]"%(val(n,CGNSNodeNameAttribute),n.nodeName)
        menu.add_command(label=s,font=cgtTreeDefaultFont)
        menu.add('separator')
        if (shwa and self.id[0].attributes):
            (dval, dids, vmax)=self.getAttributeList()
            for k in dval.keys():
                fs="%%%ds : %%s"%vmax
                s=fs%(k,dval[k])
                menu.add_command(label=s,font=dft,command=self.changeAtt)      
        menu.tk_popup(event.x_root, event.y_root)

    def popup_menu3(self, event):
      # empty file
      if (self.id[0] == None): return
      n=self.id[0]
      dft=cgtTreeDefaultFont
      shwa=self.widget.top.showattributes.get()
      #shwa=1 # force display for this menu
      #menu=Menu(self.widget, tearoff=0)
      #if (GlobalSchemaGrammar.grammar.has_key(n.nodeName)):
      #  for k in GlobalSchemaGrammar.grammar[n.nodeName]:
      #     menu.add_command(label=k,font=dft,command=self.changeAtt)      
      #menu.tk_popup(event.x_root, event.y_root)

    def expandNode(self,event):
      self.widget.top.status.grab_set()
      if self.expandable_flag:
        if self.expanded_flag:
          self.PVT_set_state(0)
        else:
          if (not self.widget.top.expandall.get()):
              self.expand()
          else: self.widget.top.frame.expandSubTree()
      self.widget.top.status.grab_release()
      self.focusNode(event)

    def focusNode(self,event):
      self.PVT_enter(event)
      self.widget.top.frame.move_cursor(self.widget.target,mark=1)

    #def nothing(self):
    #  pass

    def changeAtt(self):
      pass

    def editNode(self):
      pass

    def pasteAsBrotherNode(self):
      self.widget.pasteSubTree(self.id[0],
                               brother=1,
                               clear=self.widget.top.clearBuffer.get())
      self.widget.updateView(self)
        
    def pasteAsChildNode(self):
      self.widget.pasteSubTree(self.id[0],
                               brother=0,
                               clear=self.widget.top.clearBuffer.get())
      self.widget.updateView(self)
        
    def copyNode(self):
      self.widget.copySubTree(self.id[0])
      self.widget.target.widget.itemconfig(self.widget.target.label,fill='red')
      
    def importNode(self):
      pass

    def exportNode(self):
      pass

    def markNode(self,event):
      self.widget.target.widget.itemconfig(self.widget.target.label,fill='red')

    def linkNode(self):
      pass

    def deleteNode(self):
      self.widget.deleteSubTree(self.id[0])
      self.widget.updateView(self)

    def n_ntype(self):
      pass

    def n_dtype(self):
      pass

    def getAttributeList(self,id=None):
      if (not id): id=self.id[0]
      vmax=0
      dval={}
      dids={}
      for k in id.attributes.keys():
        if (k != 'Name'):
          att=k
          vmax=max(vmax,len(att))
          dval[att]=id.attributes[k].nodeValue              
          dids[att]=id.attributes[k]
      return (dval, dids, vmax)

    # ------------------------------------------------------------
    def getContents(self):
      # empty file
      if (self.id[0] == None): return
      #
      shwa=self.widget.top.showattributes.get()
      # show attributes (or not)
      if (shwa and self.id[0].attributes):
        (dval, dids, vmax)=self.getAttributeList()
        for k in dval.keys():
          fs="%%%ds : %%s"%vmax
          s=fs%(k,dval[k])
          nn=self.widget.add_node(name=s,
                                  collapsed_icon=self.attribute_icon,
                                  id=(dids[k],
                                      cgtTreeDefaultFontAttribute,
                                      k,
                                      None),
                                  flag=0)
      # show children
      for cnode in self.id[0].childNodes:
        if (cnode.nodeType == XmlDom.Node.COMMENT_NODE):
          # Have to take care about comments ?
          if (self.widget.top.showcomment.get()):
            ft=cgtTreeDefaultFontComment
            self.widget.add_node(name="Comment",
                                 collapsed_icon=self.comment_icon,
                                 id=(cnode,ft),
                                 flag=0)
          
        elif (cnode.nodeType == XmlDom.Node.TEXT_NODE):
          # get text, eliminate stuff like \10,\13, tab, spaces...
          # get the first one anyway !
          tt=cnode.data
          if (tt and (not self.re_empty.match(tt))):
            shwdata=self.widget.top.showdata.get()
            folder=0
            name="Data"
            ft=cgtTreeDefaultFont
            if (shwdata):
              self.widget.add_node(name=name,id=(cnode,ft),flag=folder)
        else:
          folder=1
          name="%s"%(val(cnode,CGNSNodeNameAttribute))
          ft=cgtTreeDefaultFont
          if (name in CGNSNameList): #+GlobalSchemaGrammar.grammar.keys()):
            ft=cgtTreeDefaultFont+('bold',)
          self.widget.add_node(name=name,
                     expanded_icon=self.open_icon,
                     collapsed_icon=self.shut_icon,
                     id=(cnode,ft),
                     flag=folder)


# ------------------------------------------------------------
def at(attname):
  return (None, u'%s'%attname)

# ------------------------------------------------------------
def val(node,att):
  try:
    return node.attributes[at(att)].nodeValue
  except KeyError:
    return None
  except TypeError:
    return None
  
# ------------------------------------------------------------
class cgtTree(Tree.Tree):
    def __init__(self, master, schema, file, *args, **kw_args):
      self.topwidget=master
      self.statusDisplay=0
      self.root_id=None
      self.args=args
      self.file=file
      self.kw_args=kw_args
      self.xml_dom_schema=None # means default
#      if (schema):
#        readerS = expat.Reader()
#        self.xml_dom_schema=readerS.fromUri(schema)
#        self.schemaTree=cgtParseSchema.Schema(self.xml_dom_schema)
#        self.schemaTree.parse()
#        #print self.schemaTree.grammar
#        #self.schemaTree=cgtParseSchema.Schema(schema)
      global GlobalSchemaGrammar
      #import TestSchema
      #GlobalSchemaGrammar=TestSchema.schemaTree
      if (self.file):
        self.xml_dom_object=XmlDom.parse(self.file)
#        print self.xml_dom_object.childNodes
        self.root_id=self.xml_dom_object.childNodes[0]
        # Skip HEADER comment only
        if (self.root_id.nodeType == XmlDom.Node.COMMENT_NODE):
          self.root_id=self.xml_dom_object.childNodes[1]
      self.kw_args['get_contents_callback']=CGNSNode.getContents
      self.kw_args['node_class']=CGNSNode
      self.kw_args['outlinecolor']='Coral'
      self.kw_args['expanded_icon']=PhotoImage(data=baseIcon.expanded_original)
      self.kw_args['collapsed_icon']=PhotoImage(data=baseIcon.collapsed_original)
      self.kw_args['regular_icon']=PhotoImage(data=baseIcon.regular_original)
      self.kw_args['plus_icon']=PhotoImage(data=baseIcon.plus_original)
      self.kw_args['minus_icon']=PhotoImage(data=baseIcon.minus_original)
      if (self.file):
        apply(Tree.Tree.__init__,(self,master,(self.root_id,cgtTreeDefaultFont))+self.args,self.kw_args)
      else:
        apply(Tree.Tree.__init__,(self,master,(None,cgtTreeDefaultFont))+self.args,self.kw_args)
    def getPath(self,node,pth=""):
        if (pth): pth = '/'+pth
        # attributes last item is None
        if (node.parent_node and node.id[-1]):
          if val(node.id[0],CGNSNodeNameAttribute):
            pth=val(node.id[0],CGNSNodeNameAttribute)+pth
          pth=self.getPath(node.parent_node,pth=pth)
        elif (node.parent_node and not node.id[-1]): 
          pth="[%s]"%node.id[2]+pth # should be a leaf
          pth=self.getPath(node.parent_node,pth=pth)
        return pth
    def getLevel(self,node):
        pth=self.getPath(node)
        if (not pth): return 0
        level=len(string.split(pth,'/'))
        if (pth[0] == '/'): level=level-1
        return level
    def move_cursor(self, node, mark=0):
        """Move cursor to node"""
        self.pos=node
        if (self.statusDisplay):
          self.topwidget.status.set("%s",self.getPath(node))
        x1, y1, x2, y2=self.bbox(node.symbol, node.label)
        if (mark): self.coords(self.cursor_box, x1-1, y1-1, x2+1, y2+1)
        self.see(node.symbol, node.label)
        self.update()
    def expandAllTree(self):
        p=None
        self.first()
        while (p != self.cursor_node(None)):
          p=self.cursor_node(None)
          self.descend(move=0)
        self.first()
        start=self.cursor_node(None)
        self.move_cursor(start,mark=1)
    def expandSubTree(self):
        p=None
        start=self.cursor_node(None)
        level=self.getLevel(start)
        while (p != self.cursor_node(None)):
          p=self.cursor_node(None)
          self.descend(move=0)
          if (self.getLevel(self.cursor_node(None)) <= level): break
        self.move_cursor(start)
    def updateView(self,node):
      print "UPDATE" # how ?

    # --------------------------------------------------
    def copySubTree(self,id):
      print "COPY:", id
      self.bufferSubTree=id.cloneNode(999)
    def deleteSubTree(self,id):
      print "DELETE:", id
      self.bufferSubTree=id.parentNode.removeChild(id)
    def pasteSubTree(self,id,brother,clear):
      if (not self.bufferSubTree): return
      tpst="CHILD"  
      if (brother): tpst="BROTHER"
      print "PASTE as %s:%s"%(tpst,id)
      if (not clear): nextbufferSubTree=self.bufferSubTree.cloneNode(999)
      if (not brother):
        id.appendChild(self.bufferSubTree)
      else:
        id.parentNode.appendChild(self.bufferSubTree)
      if (clear): self.bufferSubTree=None
      else:       self.bufferSubTree=nextbufferSubTree
