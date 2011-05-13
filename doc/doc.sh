mkdir build/doc 2>/dev/null
mkdir build/doc/pdf 2>/dev/null
mkdir build/doc/html 2>/dev/null
mkdir build/doc/html/_pdf 2>/dev/null

#WITHPDFFILES="YES BOY YOU GO"
#WEBSITEUPDATE="OH OH OH... HERE WE GO"
#WEBSITEUPDATESTARTSSH="LONG TIME YOUVE SEEN ME"

do_mod()
{
 export PYCGNSMOD=$1
 export PYCGNSDOC=$2
 sphinx-build -n -b html -c doc $3 build/doc/html$5

 if test "x${WITHPDFFILES}" != "x"
 then
  sphinx-build -b latex -c doc $3 build/doc/latex$5
  (cd build/doc/latex$5; pdflatex pyCGNS_$4.tex)
 fi
}

do_mod intro index doc/Intro intro ' '
do_mod PAT readme PATternMaker PAT /PAT
do_mod NAV readme NAVigater NAV /NAV
do_mod MAP readme MAPper MAP /MAP
do_mod DAT readme DATaTracer DAT /DAT
do_mod WRA readme WRApper WRA /WRA
do_mod APP readme APPlicater APP /APP
do_mod VAL readme VALidater VAL /VAL

# --- ALL
cp build/doc/latex/*.pdf   ./build/doc/pdf
cp build/doc/latex/*/*.pdf ./build/doc/pdf
cp build/doc/pdf/* ./doc
cp build/doc/pdf/* ./build/doc/html/_pdf
mkdir build/doc/html/images 2>/dev/null
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
cd /home/project-web/pycgns/htdocs
tar xvf pyCGNS-html.tar
exit
EOC

fi
# ---

