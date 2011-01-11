mkdir build/doc 2>/dev/null
mkdir build/doc/pdf 2>/dev/null
mkdir build/doc/html 2>/dev/null
mkdir build/doc/html/_pdf 2>/dev/null

WITHPDFFILES="YES BOY YOU GO"
WEBSITEUPDATE="OH OH OH... HERE WE GO"
#WEBSITEUPDATESTARTSSH="LONG TIME YOUVE SEEN ME"

# gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER \
#    -dFirstPage=9 -dLastPage=17 \
#    -sOutputFile=SIDS-to-Python.pdf pyCGNS_MAP.pdf 


# --- intro
export PYCGNSMOD='intro'
export PYCGNSDOC='index'
sphinx-build -b html -c doc doc/Intro build/doc/html

if test "x${WITHPDFFILES}" != "x"
then
 sphinx-build -b latex -c doc doc/Intro build/doc/latex
 (cd build/doc/latex; pdflatex pyCGNS_intro.tex; mv *.pdf ./../pdf)
fi

# --- PAT
export PYCGNSMOD='PAT'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc PATternMaker build/doc/html/PAT

if test "x${WITHPDFFILES}" != "x"
then
 sphinx-build -b latex -c doc PATternMaker build/doc/latex/PAT
 (cd build/doc/latex/PAT; pdflatex pyCGNS_PAT.tex; mv *.pdf ../../pdf)
fi

# --- NAV
export PYCGNSMOD='NAV'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc NAVigater build/doc/html/NAV

if test "x${WITHPDFFILES}" != "x"
then
sphinx-build -b latex -c doc NAVigater build/doc/latex/NAV
(cd build/doc/latex/NAV; pdflatex pyCGNS_NAV.tex; mv *.pdf ../../pdf)
fi

# --- MAP
export PYCGNSMOD='MAP'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc MAPper build/doc/html/MAP

if test "x${WITHPDFFILES}" != "x"
then
sphinx-build -b latex -c doc MAPper build/doc/latex/MAP
(cd build/doc/latex/MAP; pdflatex pyCGNS_MAP.tex; mv *.pdf ../../pdf)
fi

# --- DAT
export PYCGNSMOD='DAT'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc DATaTracer build/doc/html/DAT

if test "x${WITHPDFFILES}" != "x"
then
sphinx-build -b latex -c doc DATaTracer build/doc/latex/DAT
(cd build/doc/latex/DAT; pdflatex pyCGNS_DAT.tex; mv *.pdf ../../pdf)
fi

# --- WRA
export PYCGNSMOD='WRA'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc WRApper build/doc/html/WRA

if test "x${WITHPDFFILES}" != "x"
then
sphinx-build -b latex -c doc WRApper build/doc/latex/WRA
(cd build/doc/latex/WRA; pdflatex pyCGNS_WRA.tex; mv *.pdf ../../pdf)
fi

# --- APP
export PYCGNSMOD='APP'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc APPlicater build/doc/html/APP

if test "x${WITHPDFFILES}" != "x"
then
sphinx-build -b latex -c doc APPlicater build/doc/latex/APP
(cd build/doc/latex/APP; pdflatex pyCGNS_APP.tex; mv *.pdf ../../pdf)
fi

# --- ALL
cp build/doc/pdf/* ./doc
cp build/doc/pdf/* ./build/doc/html/_pdf
mkdir build/doc/html/images
cp doc/images/* build/doc/html/images
(cd build/doc/html; tar cvf ../pyCGNS-html.tar .)

# --- web site update
#
if test "x$WEBSITEUPDATE" != "x"
then
#  (cd build/doc/html;  scp -r . pycgns@pycgnsdoc2:/home/pycgns/public_html)
cd build/doc
sftp poinot,pycgns@web.sourceforge.net <<EOC   
cd htdocs
put pyCGNS-html.tar
exit
EOC

if test "x$WEBSITEUPDATESTARTSSH" != "x"
then
 ssh poinot,pycgns@shell.sourceforge.net create
fi

ssh poinot,pycgns@shell.sourceforge.net <<EOC
cd /home/groups/p/py/pycgns/htdocs
tar xvf pyCGNS-html.tar
exit
EOC

fi
# ---

