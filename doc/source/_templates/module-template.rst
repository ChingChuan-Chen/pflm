{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-undoc-members:
   :no-inherited-members:

{% if functions %}
Functions
---------

.. autosummary::
   :toctree:

{% for item in functions %}
   {{ item }}
{% endfor %}
{% endif %}

{% if classes %}
Classes
-------

.. autosummary::
   :toctree:
   :template: class-template.rst

{% for item in classes %}
   {{ item }}
{% endfor %}
{% endif %}
