mkdir build/doc 2>/dev/null
mkdir build/doc/pdf 2>/dev/null
mkdir build/doc/html 2>/dev/null
mkdir build/doc/html/_pdf 2>/dev/null

STOPREBUILDINGDOCS="TOO LONG FOR ME"
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

if test "x$STOPREBUILDINGDOCS" != "x"
then
do_mod intro index doc/Intro intro ' '
do_mod PAT _index doc/mods/PAT PAT /PAT
do_mod NAV _index doc/mods/NAV NAV /NAV
do_mod MAP _index doc/mods/MAP MAP /MAP
do_mod DAT _index doc/mods/DAT DAT /DAT
do_mod APP _index doc/mods/APP APP /APP
do_mod VAL _index doc/mods/VAL VAL /VAL
fi

# --- ALL
if test "x$WITHPDFFILES" != "x"
then
cp build/doc/latex/*.pdf   ./build/doc/pdf
cp build/doc/latex/*/*.pdf ./build/doc/pdf
cp build/doc/pdf/* ./doc
cp build/doc/pdf/* ./build/doc/html/_pdf
fi
mkdir build/doc/html/images 2>/dev/null
cp doc/images/* build/doc/html/images
(cd build/doc/html; tar cvf ../pyCGNS-html.tar .) 1>/dev/null

# --- web site update
#
if test "x$WEBSITEUPDATE" != "x"
then
#  (cd build/doc/html;  scp -r . pycgns@pycgnsdoc2:/home/pycgns/public_html)
cd build/doc
sftp poinot,pycgns@github.com <<EOC   
cd htdocs
put pyCGNS-html.tar
exit
EOC

if test "x$WEBSITEUPDATESTARTSSH" != "x"
then
 ssh poinot,pycgns@github.com create
fi

ssh poinot,pycgns@github.com <<EOC
cd /home/project-web/pycgns/htdocs
tar xvf pyCGNS-html.tar
exit
EOC

fi
# ---

