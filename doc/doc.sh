mkdir build/doc 2>/dev/null
mkdir build/doc/pdf 2>/dev/null

# --- intro
export PYCGNSMOD='intro'
export PYCGNSDOC='index'
sphinx-build -b html -c doc doc/Intro build/doc/html
sphinx-build -b latex -c doc doc/Intro build/doc/latex
(cd build/doc/latex; pdflatex pyCGNS_intro.tex; mv *.pdf ./../pdf)

# --- PAT
export PYCGNSMOD='PAT'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc PATternMaker build/doc/html/PAT
sphinx-build -b latex -c doc PATternMaker build/doc/latex/PAT
(cd build/doc/latex/PAT; pdflatex pyCGNS_PAT.tex; mv *.pdf ../../pdf)

# --- NAV
export PYCGNSMOD='NAV'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc NAVigater build/doc/html/NAV
sphinx-build -b latex -c doc NAVigater build/doc/latex/NAV
(cd build/doc/latex/NAV; pdflatex pyCGNS_NAV.tex; mv *.pdf ../../pdf)

# --- MAP
export PYCGNSMOD='MAP'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc MAPper build/doc/html/MAP
sphinx-build -b latex -c doc MAPper build/doc/latex/MAP
(cd build/doc/latex/MAP; pdflatex pyCGNS_MAP.tex; mv *.pdf ../../pdf)

# --- WRA
export PYCGNSMOD='WRA'
export PYCGNSDOC='readme'
sphinx-build -b html -c doc WRApper build/doc/html/WRA
sphinx-build -b latex -c doc WRApper build/doc/latex/WRA
(cd build/doc/latex/WRA; pdflatex pyCGNS_WRA.tex; mv *.pdf ../../pdf)
