py.install_sources([
  '__init__.py',
  'cgnsclass.py',
  'cgnserrors.py',
  'cgnskeywords.py',
  'cgnslib.py',
  'cgnstypes.py',
  'cgnsutils.py'],
  subdir: 'CGNS/PAT'
)
subdir('SIDS')
install_subdir('test', install_dir: cgns_dir / 'PAT', install_tag: 'python-runtime')
