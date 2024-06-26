"""
File names for configuration files.

Configuration files should be of the format #{NAME}.py
"""

import importlib
import inspect

PACKAGE = __package__
NAMES = [
    'params',
    'run',
]


def load_conf_module(name, key=None):
    """ Load a module and its values into globals. """

    if key:
        namespace = globals().setdefault(key.upper(), {})
    else:
        namespace = globals()

    module = importlib.import_module("%s.%s" % (PACKAGE, name))

    for (k, v) in inspect.getmembers(module):
        if k.isupper():
            if isinstance(v, str):
                namespace[k] = v.format(**globals())
            else:
                namespace[k] = v


for n in NAMES:
    # Load defaults
    load_conf_module('default.{0}'.format(n), key=n)

    # Try to load custom configuration.
    try:
        load_conf_module(n, key=n)
    except ImportError:
        pass
