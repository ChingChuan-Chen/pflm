utils_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
    fs.copyfile('__init__.pxd'),
    fs.copyfile('blas_helper.pxd'),
    fs.copyfile('lapack_helper.pxd'),
    fs.copyfile('trapz.pxd')
]

utils_extension_metadata = {
    'blas_helper': {
        'sources': [
            cython_gen_cpp.process('blas_helper.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    },
    'lapack_helper': {
        'sources': [
            cython_gen_cpp.process('lapack_helper.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    },
    'trapz': {
        'sources': [
            cython_gen_cpp.process('trapz.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    }
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
