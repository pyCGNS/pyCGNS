
cgns_dir = py.get_install_dir() / 'CGNS'

python_sources = ['__init__.py']
py.install_sources(
  python_sources,
  subdir: 'CGNS'
)

# Generate version.py for sdist
meson.add_dist_script(
   ['../_build_utils/gitversion.py', '--meson-dist', '--write',
     'CGNS/version.py']
)
if not fs.exists('version.py')
  generate_version = custom_target(
    'generate-version',
    install: true,
    build_always_stale: true,
    build_by_default: true,
    output: 'version.py',
    input: '../_build_utils/gitversion.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    install_dir: cgns_dir,
    install_tag: 'python-runtime'
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['version.py'],
    subdir : 'CGNS'
  )
endif

compilers = {
  'C': cc,
  'CYTHON': meson.get_compiler('cython')
}

machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}

conf_data = configuration_data()

# Set compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
  conf_data.set(name + '_COMP_ARGS', ', '.join(
      get_option(name.to_lower() + '_args')
    )
  )
  conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
      get_option(name.to_lower() + '_link_args')
    )
  )
endforeach

# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

conf_data.set('CROSS_COMPILED', meson.is_cross_build())

# Python information
conf_data.set('PYTHON_PATH', py.full_path())
conf_data.set('PYTHON_VERSION', py.language_version())


# Numpy include directory - needed by MAP and config
#
# this uses the path as is, and avoids running the interpreter.
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

hdf5_dep = dependency('hdf5', language: 'c')
dependency_map = {
  'HDF5': hdf5_dep,
  'NUMPY': np_dep,
}


foreach name, dep : dependency_map
  conf_data.set(name + '_NAME', dep.name())
  conf_data.set(name + '_FOUND', dep.found())
  if dep.found()
    conf_data.set(name + '_VERSION', dep.version())
    conf_data.set(name + '_TYPE_NAME', dep.type_name())
    # get_variable() results may be missing for a variety of reasons
    conf_data.set(name + '_INCLUDEDIR', dep.get_variable('includedir', default_value: 'unknown'))
    conf_data.set(name + '_LIBDIR', dep.get_variable('libdir', default_value: 'unknown'))
    conf_data.set(name + '_CONFIG', dep.get_variable('hdf5_config', default_value: 'unknown'))
    conf_data.set(name + '_PCFILEDIR', dep.get_variable('pcfiledir', default_value: 'unknown'))
  endif
endforeach

conf_data.set('PLATFORM', host_machine.system())

generate_prod_tag = run_command(
    '../_build_utils/prodtag.py',
    check: true
)
conf_data.set('DATE', generate_prod_tag.stdout().strip())



configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration : conf_data,
  install_dir: cgns_dir,
  install_tag: 'python-runtime'
)

subdir('MAP')
subdir('PAT')
subdir('VAL')

# Activate app when nav requested (dependency requirement)
if get_option('app') or get_option('nav')
    subdir('APP')
endif
if get_option('nav')
    subdir('NAV')
endif
