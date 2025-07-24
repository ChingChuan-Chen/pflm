polyfit_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
    fs.copyfile('__init__.pxd'),
    fs.copyfile('_polyfit.pxd')
]

polyfit_extension_metadata = {
    '_polyfit': {
        'sources': [
            cython_gen_cpp.process('_polyfit.pyx')
        ],
        'dependencies': [np_dep, openmp_dep],
    },
}

foreach ext_name, ext_dict : polyfit_extension_metadata
    py.extension_module(
        ext_name,
        [ext_dict.get('sources'), polyfit_cython_tree],
        dependencies: ext_dict.get('dependencies', []),
        subdir: 'pflm/smooth',
        install: true,
    )
endforeach
