
majorversion=4
minorversion=0
releaseversion=1

version="v%d.%d.%d"

include_dirs=[\
    '/home/tools/local/x86_64/include',\
    '/home/tools/local/x86_64/lib/python2.5/site-packages/numpy/core/include'\
]

library_dirs=['/home/tools/local/x86_64/lib']
optional_libs=['cgns','hdf5']

HAS_HDF5=1
