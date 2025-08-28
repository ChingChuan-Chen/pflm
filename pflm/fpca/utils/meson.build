utils_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
]

utils_extension_metadata = {
    '_raw_cov': {
        'sources': [
            cython_gen_cpp.process('_raw_cov.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    },
    'rotate_polyfit2d': {
        'sources': [
            cython_gen_cpp.process('rotate_polyfit2d.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    },
    'fpca_score': {
        'sources': [
            cython_gen_cpp.process('fpca_score.pyx'),
        ],
        'dependencies': [np_dep, openmp_dep],
    }
}

foreach ext_name, ext_dict : utils_extension_metadata
    py.extension_module(
        ext_name,
        [ext_dict.get('sources'), utils_cython_tree],
        dependencies: ext_dict.get('dependencies', []),
        subdir: 'pflm/fpca/utils',
        install: true,
    )
endforeach
