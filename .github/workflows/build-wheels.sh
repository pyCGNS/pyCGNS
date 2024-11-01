#!/bin/bash

# https://github.com/pypa/python-manylinux-demo

set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" -w wheelhouse/
    fi
}

# Compile wheels
for PYBIN in cp310-cp310 cp312-cp312; do
    "/opt/python/${PYBIN}/bin/pip" install numpy cython meson-python
    "/opt/python/${PYBIN}/bin/pip" wheel . --no-deps -w dist/
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    repair_wheel "$whl"
done
