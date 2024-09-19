from importlib import import_module


def get_module_attr(name):
    mod, attr = name.rsplit(".", 1)
    return getattr(import_module(mod), attr)
