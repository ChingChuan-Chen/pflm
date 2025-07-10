polyfit_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
    fs.copyfile('__init__.pxd'),
    fs.copyfile('_polyfit.pxd'),
]

utils_inc = include_directories('../utils/src')

polyfit_extension_metadata = {
    '_polyfit': {
        'sources': [
            cython_gen_cpp.process('_polyfit.pyx')
        ],
        'dependencies': [np_dep, openmp_dep],
        'link_with': [helper_lib],
        'include_directories': [utils_inc],
    },
}

foreach ext_name, ext_dict : polyfit_extension_metadata
    py.extension_module(
        ext_name,
        [ext_dict.get('sources'), polyfit_cython_tree],
        dependencies: ext_dict.get('dependencies', []),
        link_with: ext_dict.get('link_with', []),
        include_directories: ext_dict.get('include_directories', []),
        subdir: 'pflm/smooth',
        install: true,
    )
endforeach
