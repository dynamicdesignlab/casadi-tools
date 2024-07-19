from numbers import Real

import casadi as ca
import numpy as np

CASADI_SYMBOLIC = ca.MX | ca.SX
CASADI_TYPE = CASADI_SYMBOLIC | ca.DM
CASADI_INPUT_TYPE = CASADI_TYPE | Real

DATA_DICT = dict[str, np.ndarray]
