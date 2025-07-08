project(
  'pflm',
  'cpp', 'cython',
  version: run_command('pflm/_build_utils/version.py', check: true).stdout().strip(),
  license: 'MIT',
  meson_version: '>= 1.1.0',
  default_options: [
    'cpp_std=c++14',
  ],
)

cpp = meson.get_compiler('cpp')
cython = meson.get_compiler('cython')

py = import('python').find_installation(pure: false)

# Copy all the .py files to the install dir, rather than using
# py.install_sources and needing to list them explicitly one by one
install_subdir('pflm', install_dir: py.get_install_dir())

subdir('pflm')
