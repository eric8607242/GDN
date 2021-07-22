import sys


def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)


# Importing customizing modules
from .training_agent import TrainingAgnet
