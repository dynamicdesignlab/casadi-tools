from enum import Enum, unique
from numbers import Number
from typing import Any

import numpy as np

""" Implement functions for SI unit conversions. """

@unique
class SI_PREFIX(Enum):
    GIGA = 1e9
    MEGA = 1e6
    KILO = 1e3
    HECTO = 1e2
    DEKA = 1e1
    DECI = 1e-1
    CENT = 1e-2
    MILI = 1e-3
    MICRO = 1e-6
    NANO = 1e-9


def si_prefix_to_base_from_str(name: str, value: Any) -> tuple[Any, str]:
    """
    Convert given name and value to SI base units.

    Parameters
    ----------
    name: str
        Name of signal constaining SI prefix
    value: Any
        Numeric type

    Returns
    -------
    base_value: Any
        Value in SI base units
    base_name: str
        Name in SI base units

    """
    prefix = None
    multiplier = 1.0
    for item in SI_PREFIX:
        if item.name.lower() not in name:
            continue

        prefix = item.name.lower()
        multiplier = item.value

    if prefix is None:
        return value, name

    return value * multiplier, name.replace(prefix, "")


def si_prefix_to_base(prefix: SI_PREFIX, value: Any) -> tuple[Any, str]:
    """
    Convert given name and value to SI base units.

    Parameters
    ----------
    name: str
        Name of signal constaining SI prefix
    value: Any
        Numeric type

    Returns
    -------
    base_value: Any
        Value in SI base units
    base_name: str
        Name in SI base units

    """
    if not isinstance(prefix, SI_PREFIX):
        raise ValueError("Prefix argument must be an SI_PREFIX enum value")

    return value * prefix.value


def si_base_to_prefix(
    prefix: SI_PREFIX, value: Number | np.ndarray
) -> Number | np.ndarray:
    """
    Convert given value to SI prefix units.

    Parameters
    ----------
    name: str
        SI prefix
    value: Any
        Numeric type

    Returns
    -------
    prefix_value: Any
        Value in SI prefix units

    """
    if not isinstance(prefix, SI_PREFIX):
        raise ValueError("Prefix argument must be an SI_PREFIX enum value")

    return value / prefix.value
