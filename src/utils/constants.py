from collections import namedtuple

ParamTuple = namedtuple("ParamTuple", ["p", "gamma", "R"])
StepSizeTuple = namedtuple("StepSizeTuple", ["p", "gamma", "R"])
KnownParameter = namedtuple("KnownParameter", ["name", "value"])

StateTransition = namedtuple("StateTransition", ["s", "a", "s_next"])

p_limits = (0.5, 0.999)
gamma_limits = (0.5, 0.999)
R_limits = (0,1)

beta_agent = 20