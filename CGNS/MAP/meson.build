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

# Enable UNIX large file support on 32-bit systems (64 bit off_t,
# lseek -> lseek64, etc.)
cflags_large_file_support = []
if host_machine.system() == 'aix'
  cflags_large_file_support += '-D_LARGE_FILES'
else
  cflags_large_file_support += [
    '-D_FILE_OFFSET_BITS=64',
    '-D_LARGEFILE_SOURCE=1',
    '-D_LARGEFILE64_SOURCE=1',
  ]
endif


# NumPy include directory - needed by SidsToPython
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


# Define HDF5 dependency
# Use meson to detect headers features
hdf5_dep = dependency('hdf5', language:'c', required: true)
have_hdf5  = hdf5_dep.found()
if have_hdf5
   if cc.has_header('H5pubconf.h', dependencies: hdf5_dep)
      hdf5_h64 = 0
      hdf5_hup = 1
      hdf5_hst = 1
      hfn = 'H5pubconf.h'
   elif cc.has_header('h5pubconf.h', dependencies: hdf5_dep)
      hdf5_h64 = 0
      hdf5_hup = 0
      hdf5_hst = 1
      hfn = 'h5pubconf.h'
   elif cc.has_header('H5pubconf-64.h', dependencies: hdf5_dep)
      hdf5_h64 = 1
      hdf5_hup = 1
      hdf5_hst = 0
      hfn = 'H5pubconf-64.h'
   elif cc.has_header('h5pubconf-64.h', dependencies: hdf5_dep)
      hdf5_h64 = 1
      hdf5_hup = 0
      hdf5_hst = 0
      hfn = 'h5pubconf-64.h'
   else
      hdf5_h64 = 0
      hdf5_hup = 1
      hdf5_hst = 1
      hfn = 'H5pubconf.h'
   endif

   h5_have_parallel = cc.get_define('H5_HAVE_PARALLEL', dependencies: hdf5_dep, prefix:f'#include "@hfn@"')
   if h5_have_parallel == '' 
     h5_have_parallel = 0
   endif
else
   hdf5_h64 = 0
   hdf5_hup = 1
   hdf5_hst = 1
   h5_have_parallel = 0
endif

chlone_conf = configuration_data()

if get_option('has-pthread')
   chlone_conf.set('CHLONE_HAS_PTHREAD', 1)
   thread_dep = dependency('threads')
else
   chlone_conf.set('CHLONE_HAS_PTHREAD', 0)
   thread_dep = []
endif

if get_option('has-regexp')
   chlone_conf.set('CHLONE_HAS_REGEXP', 1)
else
   chlone_conf.set('CHLONE_HAS_REGEXP', 0)
endif

if get_option('printf-trace')
   chlone_conf.set('CHLONE_PRINTF_TRACE', 1)
else
   chlone_conf.set('CHLONE_PRINTF_TRACE', 0)
endif

if get_option('use-compact-storage')
   chlone_conf.set('CHLONE_USE_COMPACT_STORAGE', 1)
else
   chlone_conf.set('CHLONE_USE_COMPACT_STORAGE', 0)
endif

if is_windows
   chlone_conf.set('CHLONE_ON_WINDOWS', 1)
else
   chlone_conf.set('CHLONE_ON_WINDOWS', 0)
endif

if have_hdf5
   chlone_conf.set('HDF5_VERSION', hdf5_dep.version())
   chlone_conf.set('CHLONE_H5CONF_STD', hdf5_hst)
   chlone_conf.set('CHLONE_H5CONF_64',  hdf5_h64)
   chlone_conf.set('CHLONE_H5CONF_UP',  hdf5_hup)
endif
chlone_conf.set('CHLONE_INSTALL_LIBRARIES', '')
chlone_conf.set('CHLONE_INSTALL_INCLUDES', '')

src_chlone = ['l3.c',
  'error.c',
  'linksearch.c',
  'sha256.c']

src_cgns_map = ['SIDStoPython.c',]


config_h = configure_file(input:'CHLone_config.h.in',
  output: '@BASENAME@',
  format: 'cmake',
  configuration: chlone_conf,
  install: true,
  install_dir:  cgns_dir / 'MAP/include',
  install_tag: 'devel')

embedded_pyx = configure_file(input:'EmbeddedCHLone.pyx.in',
  output:'@BASENAME@',
  configuration: chlone_conf)


# Set common build flags for C code
# -------------------------------------
# Common build flags
c_args_common = [
  cflags_large_file_support,
]
# flags for hdf5
if not is_windows or is_mingw
  c_args_common += ['-Wno-return-type']
endif

if is_mingw
  c_args_common += ['-Wno-format-extra-args', '-Wno-format']
endif

# Add Macro definition for HDF5 on Windows
# _extra_macro_definition_
if is_windows
  c_args_common += ['-D_HDF5USEDLL_', '-DH5_BUILT_AS_DYNAMIC_LIB']
endif


# Targets can depend on generated headers and include directories from
# MAP, wrap those up into a reusable dependency.
map_dep = declare_dependency(
  sources: [
    config_h,
  ],
  include_directories: [
    '.',
  ]
)

# Copy the main __init__.py|pxd fie to the build dir (needed for cython)
#__init__py = fs.copyfile('__init__.py')

py.extension_module('EmbeddedCHLone', 
  [
    config_h,
    embedded_pyx,
    src_chlone,
    src_cgns_map,
  ],
  c_args: c_args_common,
  include_directories: [
   '.',
  ],
  dependencies: [np_dep, hdf5_dep, thread_dep],
  install: true,
  subdir: 'CGNS/MAP',
)

python_sources = [
  '__init__.py',
  'cgio.py',
]

py.install_sources(
  python_sources,
  subdir: 'CGNS/MAP'
)

install_subdir('test', install_dir: cgns_dir / 'MAP', install_tag: 'python-runtime')
