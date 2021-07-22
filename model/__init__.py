import sys

def get_model_class(name):
    return getattr(sys.modules[__name__], f"{name}Model")

from .gdn import GDNModel
