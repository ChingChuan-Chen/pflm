interp_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
    fs.copyfile('src/interp.h'),
]

interp_extension_metadata = {
    'interp': {
        'sources': [
            cython_gen_cpp.process('interp.pyx'),
            'src' / 'interp.cpp'
        ],
        'dependencies': [np_dep, openmp_dep],
    },
}

foreach ext_name, ext_dict : interp_extension_metadata
    py.extension_module(
        ext_name,
        [ext_dict.get('sources'), interp_cython_tree],
        dependencies: ext_dict.get('dependencies', []),
        subdir: 'pflm/interp',
        install: true,
    )
endforeach
