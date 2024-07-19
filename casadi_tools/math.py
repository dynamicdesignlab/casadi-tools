"""
Implement miscellaneous functions and types.

This module implements several useful funcitons that don't fit
well in any other module. Intended to reduce code duplication in
other modules.

"""

import casadi as ca
import numpy as np

from casadi_tools import types
from casadi_tools.nlp_utils import casadi_builder as cb


@cb.casadi_function((1,))
def wrap_to_pi(angle: types.CASADI_INPUT_TYPE) -> types.CASADI_TYPE:
    """
    Wrap angle to pi.

    Reform angle to lie between [-pi, pi]

    Parameters
    ----------
    angle: types.CASADI_INPUT_TYPE
        Angle in radians

    Returns
    -------
    wrap_angle: types.CASADI_TYPE
        Angle between [-pi, pi]

    """
    return ca.arctan2(ca.sin(angle), ca.cos(angle))


def wrap_to_pi_float(angle: float | np.ndarray) -> float | np.ndarray:
    """
    Wrap angle to pi.

    Reform angle to lie between [-pi, pi]

    Parameters
    ----------
    angle: float | np.ndarray
        Angle in radians

    Returns
    -------
    wrap_angle: float | np.ndarray
        Angle between [-pi, pi]

    """
    return np.arctan2(np.sin(angle), np.cos(angle))


@cb.casadi_function((1, 1, 1))
def clamp_val(
    val: types.CASADI_INPUT_TYPE,
    lower: types.CASADI_INPUT_TYPE,
    upper: types.CASADI_INPUT_TYPE,
) -> types.CASADI_TYPE:
    """
    Clamp input value to limits.

    Parameters
    ----------
    val: types.CASADI_INPUT_TYPE
        Value to clamp
    lower: types.CASADI_INPUT_TYPE
        Lower value limit
    upper: types.CASADI_INPUT_TYPE
        Upper value limit

    Returns
    -------
    clamp_val : types.CASADI_TYPE
        Clamped value

    """
    val = ca.fmin(val, upper)
    val = ca.fmax(val, lower)

    return val
