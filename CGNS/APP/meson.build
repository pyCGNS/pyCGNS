python_sources = [
  '__init__.py',
  ]

py.install_sources(
  python_sources,
  subdir: 'CGNS/APP'
)

subdir('lib')
subdir('examples')
subdir('misc')
install_subdir('test', install_dir: cgns_dir / 'APP', install_tag: 'python-runtime')
install_subdir('tools', install_dir: cgns_dir / 'APP', install_tag: 'python-runtime')
