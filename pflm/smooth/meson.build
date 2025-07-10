polyfit_cython_tree = [
    pflm_root_cython_tree,
    fs.copyfile('__init__.py'),
    fs.copyfile('__init__.pxd'),
    fs.copyfile('_polyfit.pxd'),
    fs.copyfile('src/polyfit_helper.h')
]

polyfit_extension_metadata = {
    '_polyfit': {
        'sources': [
            cython_gen_cpp.process('_polyfit.pyx'),
            'src' / 'polyfit_helper.cpp'
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
