# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_define('__MINGW32__') != ''

if is_mingw
  is_mingw_built_python = run_command(
    py, ['-c', 'import sysconfig; print(sysconfig.get_platform())'],
    check: true).stdout().strip().startswith('mingw')
  if not is_mingw_built_python
    # For mingw-w64, link statically against the UCRT.
    gcc_link_args = ['-lucrt', '-static']
    add_project_link_arguments(gcc_link_args, language: ['c'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
  endif
  # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
  add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c'])
endif

# https://mesonbuild.com/FAQ.html#why-does-building-my-project-with-msvc-output-static-libraries-called-libfooa)
if is_windows and cc.get_id() == 'msvc'
  name_prefix_staticlib = ''
  name_suffix_staticlib = 'lib'
else
  name_prefix_staticlib = []
  name_suffix_staticlib = []
endif

# NumPy include directory - needed by cython/pyx
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _incdir_numpy_abs = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)
# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released MAP versions
# when NumPy introduces a new deprecation.
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)

# Create a config file to tell NAV if we are on windows
#
conf_nav = configuration_data()
if is_windows
    conf_nav.set('HAS_MSW', 1)
else
    conf_nav.set('HAS_MSW', 0)
endif
if get_option('use-vtk')
    conf_nav.set('USE_VTK', 1)
else
    conf_nav.set('USE_VTK', 0)
endif

config_py = configure_file(
   input:'config.py.in',
   output: '@BASENAME@',
   configuration: conf_nav,
   install: true,
   install_dir: py.get_install_dir(subdir:'CGNS/NAV'),
   install_tag: 'python-runtime')

# Qt windows
modnamelist = ['Q7TreeWindow',
        'Q7DiffWindow',
        'Q7MergeWindow',
        'Q7ControlWindow',
        'Q7OptionsWindow',
        'Q7FormWindow',
        'Q7FileWindow',
        'Q7QueryWindow',
        'Q7SelectionWindow',
        'Q7InfoWindow',
        'Q7DiagWindow',
        'Q7LinkWindow',
        'Q7HelpWindow',
        'Q7ToolsWindow',
        'Q7PatternWindow',
        'Q7AnimationWindow',
        'Q7MessageWindow',
        'Q7LogWindow']

if get_option('use-vtk')
   modnamelist += ['Q7VTKWindow']
endif

mfile_list = [
        'mtree',
        'mquery',
        'mcontrol',
        'mtable',
        'mpattern',
        'diff',
        'mdifftreeview',
        'merge',
        'mmergetreeview',
]

if get_option('use-vtk')
  mfile_list += ['mparser']
endif

foreach mfile: mfile_list
  py.extension_module(mfile,
    [mfile+'.pyx'],
    include_directories: [inc_np],
    dependencies: [np_dep, py_dep],
    install: true,
    subdir: 'CGNS/NAV',
  )
endforeach

pyuic_list = ['pyuic', 'pyside6-uic', 'pyuic5']
pyrcc_list = ['pyrcc', 'pyside6-rcc', 'pyrcc5']
pyuic = find_program(pyuic_list, dirs:[fs.parent(py.full_path())], required:true)
pyrcc = find_program(pyrcc_list, dirs:[fs.parent(py.full_path())], required:true)

# Handle Resources
res_py = custom_target('Res',
    input: 'R/Res.qrc',
    output: 'Res_rc.py',
    command: [pyrcc, '-o', '@OUTPUT@', '@INPUT@'],
    install: true,
    install_dir: py.get_install_dir(subdir:'CGNS/NAV'),
    install_tag: 'python-runtime'
)


foreach mfile: modnamelist
  # Handle UI generation
  ui_src = custom_target(mfile+'.pyx',
    input: 'T/'+mfile+'.ui',
    output: mfile+'.pyx',
    command: [pyuic, '--from-imports', '-o', '@OUTPUT@', '@INPUT@']
    )
  py.extension_module(mfile,
    [ui_src],
    include_directories: [inc_np],
    dependencies: [np_dep, py_dep],
    install: true,
    subdir: 'CGNS/NAV',
  )
endforeach

python_sources = [
  '__init__.py',
  'moption.py',
  'pqueries.py',
  'QVTKRenderWindowInteractor.py',
  'Q7VTKRenderWindowInteractor.py',
  'script.py',
  'uqueries.py',
  'wcontrol.py',
  'wdiag.py',
  'wdifftreeview.py',
  'weditors.py',
  'wfile.py',
  'wfingerprint.py',
  'wform.py',
  'whelp.py',
  'winfo.py',
  'wlink.py',
  'wmergetreeview.py',
  'wmessages.py',
  'woption.py',
  'wpattern.py',
  'wquery.py',
  'wstylesheets.py',
  'wtools.py',
  'wtree.py',
  'wvtk.py',
  'wvtkutils.py',
]

py.install_sources(
  python_sources,
  subdir: 'CGNS/NAV'
)


install_subdir('test', install_dir: cgns_dir / 'NAV', install_tag: 'python-runtime')
install_subdir('tools', install_dir: cgns_dir / 'NAV', install_tag: 'python-runtime')

