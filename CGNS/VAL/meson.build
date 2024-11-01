python_sources = [
  '__init__.py',
  'SIDS.py',
  'simplecheck.py',
  'CGNS_VAL.py',
  ]

py.install_sources(
  python_sources,
  subdir: 'CGNS/VAL'
)

subdir('grammars')
subdir('parse')
subdir('suite')
install_subdir('test', install_dir: cgns_dir / 'VAL', install_tag: 'python-runtime')
