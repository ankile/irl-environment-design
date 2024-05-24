from collections import namedtuple

ParamTuple = namedtuple("ParamTuple", ["p", "gamma", "R"])
GenParamTuple = namedtuple("GenParamTuple", ["T", "gamma", "R"])

StepSizeTuple = namedtuple("StepSizeTuple", ["p", "gamma", "R"])
KnownParameter = namedtuple("KnownParameter", ["name", "value"])

StateTransition = namedtuple("StateTransition", ["s", "a", "s_next"])

p_limits = (0.5, 0.95)
gamma_limits = (0.5, 0.95)
R_limits = (0,1)

beta_agent = 20


# Import enum types
from enum import Enum


class Action(Enum):
    """Action types for the agent"""

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
