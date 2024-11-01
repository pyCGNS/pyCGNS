python_sources = [
  '__init__.py',
  'addNodes.py',
  'getBCbyFamily.py',
  'map_loadandsave.py',
  'merge.py',
  'T0.py',
  ]

py.install_sources(
  python_sources,
  subdir: 'CGNS/APP/examples'
)

install_subdir('MAP', install_dir: cgns_dir / 'APP/examples', install_tag: 'python-runtime')
install_subdir('PAT', install_dir: cgns_dir / 'APP/examples', install_tag: 'python-runtime')
