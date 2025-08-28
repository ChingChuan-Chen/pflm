# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
import inspect
import re
from datetime import datetime
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Add project root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'pflm'
author = 'Ching-Chuan Chen'
copyright = f'{datetime.now().year}, {author}'
# Get version without importing the package (avoid importing C extensions)
try:
    release = _pkg_version("pflm")
except PackageNotFoundError:
    release = os.environ.get('PFLM_VERSION', '0.1.0.dev0')
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', release)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
print(f"{version} {release}")

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',
    'numpydoc',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build']
language = 'en'

# Let autosummary generate stubs automatically
autosummary_generate = True
# Include members re-exported in __init__.py
autosummary_imported_members = True

# Show short names (hide module prefixes)
add_module_names = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': False,
    'show-inheritance': True,
    'class-doc-from': 'class',
}

autodoc_typehints = 'signature'
autodoc_typehints_format = 'short'
autoclass_content = 'class'

numpydoc_show_class_members = False

# Mock C/Cython extensions during docs build
autodoc_mock_imports = [
    'pflm.interp.interp',
    'pflm.smooth.polyfit',
    'pflm.fpca.utils._raw_cov',
    'pflm.fpca.utils.fpca_score',
    'pflm.fpca.utils.rotate_polyfit2d',
    'pflm.utils.blas_helper',
    'pflm.utils.lapack_helper',
    'pflm.utils.trapz',
]

# Intersphinx cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

# -- HTML --------------------------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'logo': {
        'text': 'pflm',
    },
    'show_prev_next': False,
    'use_edit_page_button': True,
    'external_links': [
        {'name': 'GitHub', 'url': 'https://github.com/ChingChuan-Chen/pflm'},
    ],
}
html_context = {
    'github_user': 'ChingChuan-Chen',
    'github_repo': 'pflm',
    'github_version': 'main',
    'doc_path': 'doc/source',
}

# Generate GitHub source links (NumPy-like linkcode)


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    modname = info.get('module')
    fullname = info.get('fullname')
    if not modname:
        return None
    try:
        module = __import__(modname, fromlist=['*'])
    except Exception:
        return None

    obj = module
    for part in (fullname or '').split('.'):
        if not part:
            continue
        if not hasattr(obj, part):
            obj = None
            break
        obj = getattr(obj, part)

    # Locate source file and line range
    try:
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        try:
            fn = inspect.getsourcefile(module) or inspect.getfile(module)
            source, lineno = inspect.getsourcelines(module)
        except Exception:
            return None

    # Only link to files within the project tree
    fn = os.path.relpath(fn, start=os.path.abspath('../../'))
    if fn.startswith('..'):
        return None

    end = lineno + len(source) - 1
    return f"https://github.com/ChingChuan-Chen/pflm/blob/main/{fn}#L{lineno}-L{end}"
