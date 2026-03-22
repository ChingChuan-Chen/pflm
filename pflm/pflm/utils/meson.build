pflm_utils_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py')
]

pflm_utils_extension_metadata = {
    'elastic_net_solver': {
        'sources': [
            cython_gen_cpp.process('elastic_net_solver.pyx')
        ],
        'dependencies': [np_dep, openmp_dep],
    },
}

foreach ext_name, ext_dict : pflm_utils_extension_metadata
    py.extension_module(
        ext_name,
        [ext_dict.get('sources'), pflm_utils_cython_tree],
        dependencies: ext_dict.get('dependencies', []),
        subdir: 'pflm/pflm/utils',
        install: true,
    )
endforeach
