import sys, os

extensions = ['sphinx.ext.autodoc','sphinx.ext.intersphinx']
mapdir='%s/../build/doc/html'%os.path.abspath('.')
intersphinx_mapping={'wraix': (mapdir+'/WRA',None),
                     'mapix': (mapdir+'/MAP',None),
                     'patix': (mapdir+'/PAT',None),
                     'navix': (mapdir+'/NAV',None),
                     'appix': (mapdir+'/APP',None),
                     'valix': (mapdir+'/VAL',None),
                     'datix': (mapdir+'/DAT',None)
                     }

#templates_path = ['_templates']
html_theme_path = ['.']
source_suffix = '.txt'

import os
master_doc=os.environ['PYCGNSDOC']
project = u'pyCGNS'
copyright = u'2010, Marc Poinot'
version = '4'
release = '4.0.1'
unused_docs = ['license.txt']
exclude_trees = ['VALidater','TRAnslater','DATaTracer','APPlicater',
                 'build','doc','lib','.hg']
exclude_dirnames = ['VALidater','TRAnslater','DATaTracer','APPlicater',
                 'build','doc','lib','.hg']

pygments_style = 'sphinx'
html_theme = 'pycgns'
html_title = "%s"%os.environ['PYCGNSMOD']
html_logo = 'images/%s-logo-small.jpg'%os.environ['PYCGNSMOD']
html_favicon =  'images/pyCGNS-logo-tiny.ico'
#html_static_path = ['_static']
html_use_index = True
htmlhelp_basename = 'pyCGNSdoc'

latex_paper_size = 'a4'
latex_font_size = '10pt'
latex_documents = [
  (os.environ['PYCGNSDOC'],
   'pyCGNS_%s.tex'%os.environ['PYCGNSMOD'],
   u'pyCGNS.%s/Manual'%os.environ['PYCGNSMOD'],
   u'Marc Poinot',
   'manual',False),
]
latex_logo = 'images/%s-logo.jpg'%os.environ['PYCGNSMOD']
latex_use_parts = False
latex_use_modindex = True

autodoc_member_order='bysource'

# --- last line
