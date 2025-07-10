utils_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
    fs.copyfile('__init__.pxd'),
    fs.copyfile('_lapack_helper.pxd'),
    fs.copyfile('src/helper.h'),
]

helper_lib = static_library(
    'helper',
    'src/helper.cpp',
    include_directories: include_directories('src'),
    dependencies: [openmp_dep],
    install: false
)

utils_extension_metadata = {
    '_lapack_helper': {
        'sources': [
            cython_gen_cpp.process('_lapack_helper.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    },
}

foreach ext_name, ext_dict : utils_extension_metadata
    py.extension_module(
        ext_name,
        [ext_dict.get('sources'), utils_cython_tree],
        dependencies: ext_dict.get('dependencies', []),
        subdir: 'pflm/utils',
        install: true,
    )
endforeach
