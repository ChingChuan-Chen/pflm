fs = import('fs')

cython_args = []

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cpp.get_id() == 'gcc'

# Adapted from scipy, each project seems to have its own tweaks for this. One
# day using dependency('numpy') will be a thing, see
# https://github.com/mesonbuild/meson/issues/9598.
# NumPy include directory - needed in all submodules
# Relative paths are needed when for example a virtualenv is
# placed inside the source tree; Meson rejects absolute paths to places inside
# the source tree. The try-except is needed because when things are split
# across drives on Windows, there is no relative path and an exception gets
# raised. There may be other such cases, so add a catch-all and switch to
# an absolute path.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
    incdir_numpy = run_command(
        py,
        [
            '-c',
            '''
import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
''',
        ],
        check: true,
    ).stdout().strip()
endif

inc_np = include_directories(incdir_numpy)
# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation.
numpy_no_deprecated_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(
    include_directories: inc_np,
    compile_args: numpy_no_deprecated_api,
)

openmp_dep = dependency('OpenMP', language: 'cpp', required: false)

if not openmp_dep.found()
    warn_about_missing_openmp = true
    # On Apple Clang avoid a misleading warning if compiler variables are set.
    # See https://github.com/scikit-learn/scikit-learn/issues/28710 for more
    # details. This may be removed if the OpenMP detection on Apple Clang improves,
    # see https://github.com/mesonbuild/meson/issues/7435#issuecomment-2047585466.
    if host_machine.system() == 'darwin' and cpp.get_id() == 'clang'
        compiler_env_vars_with_openmp = run_command(
            py,
            [
                '-c',
                '''
import os

compiler_env_vars_to_check = ["CPPFLAGS", "CFLAGS", "CXXFLAGS"]

compiler_env_vars_with_openmp = [
    var for var in compiler_env_vars_to_check if "-fopenmp" in os.getenv(var, "")]
print(compiler_env_vars_with_openmp)
''',
            ],
            check: true,
        ).stdout().strip()
        warn_about_missing_openmp = compiler_env_vars_with_openmp == '[]'
    endif
    if warn_about_missing_openmp
        warning(
            '''
                ***********
                * WARNING *
                ***********

It seems that pflm cannot be built with OpenMP.

- Make sure you have followed the installation instructions:

    https://scikit-learn.org/dev/developers/advanced_installation.html

- If your compiler supports OpenMP but you still see this
  message, please submit a bug report at:

    https://github.com/scikit-learn/scikit-learn/issues

- The build will continue with OpenMP-based parallelism
  disabled. Note however that some estimators will run in
  sequential mode instead of leveraging thread-based
  parallelism.

                    ***
''',
        )
    else
        warning(
            '''It looks like compiler environment variables were set to enable OpenMP support.
Check the output of "import pflm; pflm.show_versions()" after the build
to make sure that pflm was actually built with OpenMP support.
''',
        )
    endif
endif

pflm_cython_args = [
    '-X language_level=3',
    '-X boundscheck=False',
    '-X wraparound=False',
    '-X initializedcheck=False',
    '-X nonecheck=False',
    '-X cdivision=True',
    '-X profile=False',
    '--include-dir',
    meson.global_build_root(),
]
cython_args += pflm_cython_args

cython_program = find_program(cython.cmd_array()[0])

cython_gen_cpp = generator(
    cython_program,
    arguments: cython_args + ['--cplus', '@INPUT@', '--output-file', '@OUTPUT@'],
    output: '@BASENAME@.cpp',
)

tempita_cli = find_program('_build_utils/tempita.py')

pflm_root_cython_tree = [fs.copyfile('__init__.py')]

pflm_dir = py.get_install_dir() / 'pflm'

subdir('interp')
