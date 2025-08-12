utils_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
    fs.copyfile('__init__.pxd'),
    fs.copyfile('_lapack_helper.pxd'),
]

utils_extension_metadata = {
    '_lapack_helper': {
        'sources': [
            cython_gen_cpp.process('_lapack_helper.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    },
    '_raw_cov': {
        'sources': [
            cython_gen_cpp.process('_raw_cov.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    },
    '_rotate_polyfit2d': {
        'sources': [
            cython_gen_cpp.process('_rotate_polyfit2d.pyx'),
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
