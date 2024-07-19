"""
Implement integrator functions and builders.

This module implements integrators for use with dynamic models built
with the casadi_tools framework.

"""

import functools as ftool
from numbers import Real
from typing import Callable

import casadi as ca

from casadi_tools import types
from casadi_tools.nlp_utils import casadi_builder as cb

OracleProtocol = Callable[
    [types.CASADI_INPUT_TYPE, types.CASADI_INPUT_TYPE, types.CASADI_INPUT_TYPE],
    types.CASADI_TYPE,
]


def euler(
    oracle: OracleProtocol,
    param0: types.CASADI_INPUT_TYPE,
    state0: types.CASADI_INPUT_TYPE,
    input0: types.CASADI_INPUT_TYPE,
    step: Real,
):
    """
    Implement simple Euler integration scheme.

    Returns
    -------
    new_state: types.CASADI_TYPE
        New dynamic states after integration performed

    Parameters
    ----------
    oracle: OracleProtocol
    param0: types.CASADI_INPUT_TYPE
        Parameters
    state0: types.CASADI_INPUT_TYPE
        Initial state
    input0: types.CASADI_INPUT_TYPE
        Initial input
    step: Real
        Integration step

    """
    return state0 + step * oracle(param0, state0, input0)


def rk2(
    oracle: OracleProtocol,
    param0: types.CASADI_INPUT_TYPE,
    state0: types.CASADI_INPUT_TYPE,
    input0: types.CASADI_INPUT_TYPE,
    step: Real,
):
    """
    Implement 2nd order Runga-Kutta integration scheme.

    Returns
    -------
    new_state: NamedVector
        New dynamic states after integration performed

    Parameters
    ----------
    oracle: OracleProtocol
        Dynamic model taking a state NamedVector and an input NamedVector
        (This must be a casadi_method or casadi_function!)
    param0: types.CASADI_INPUT_TYPE
        Parameters
    state0: types.CASADI_INPUT_TYPE
        Initial state
    input0: types.CASADI_INPUT_TYPE
        Initial input
    step: Real
        Integration step

    """
    step1 = step * oracle(param0, state0, input0)
    step2 = step * oracle(param0, state0 + step1 / 2.0, input0)
    return state0 + step2


def rk4(
    oracle: OracleProtocol,
    param0: types.CASADI_INPUT_TYPE,
    state0: types.CASADI_INPUT_TYPE,
    input0: types.CASADI_INPUT_TYPE,
    step: Real,
):
    """
    Implement 2nd order Runga-Kutta integration scheme.

    Returns
    -------
    new_state: NamedVector
        New dynamic states after integration performed

    Parameters
    ----------
    oracle: OracleProtocol
        Dynamic model taking a state NamedVector and an input NamedVector
        (This must be a casadi_method or casadi_function!)
    param0: types.CASADI_INPUT_TYPE
        Parameters
    state0: types.CASADI_INPUT_TYPE
        Initial state
    input0: types.CASADI_INPUT_TYPE
        Initial input
    step: Real
        Integration step

    """
    step1 = step * oracle(param0, state0, input0)
    step2 = step * oracle(param0, state0 + step1 / 2.0, input0)
    step3 = step * oracle(param0, state0 + step2 / 2.0, input0)
    step4 = step * oracle(param0, state0 + step3, input0)
    return state0 + ((step1 + 2.0 * step2 + 2.0 * step3 + step4) / 6.0)


def trapz(
    oracle: OracleProtocol,
    param0: types.CASADI_INPUT_TYPE,
    param1: types.CASADI_INPUT_TYPE,
    state0: types.CASADI_INPUT_TYPE,
    state1: types.CASADI_INPUT_TYPE,
    input0: types.CASADI_INPUT_TYPE,
    input1: types.CASADI_INPUT_TYPE,
    step: Real,
):
    """
    Implement implicit trapezoidal integration scheme.

    Returns
    -------
    new_state: NamedVector
        New dynamic states after integration performed

    Parameters
    ----------
    oracle: OracleProtocol
        Dynamic model taking a state NamedVector and an input NamedVector
        (This must be a casadi_method or casadi_function!)

    param0: NamedVector
        Initial parameters
    param1: NamedVector
        Final parameters
    state0: NamedVector
        Initial state
    state1: NamedVector
        Final state
    input0: NamedVector
        Initial input
    input1: NamedVector
        Final input
    step: Real
        Integration step

    """
    left = oracle(param0, state0, input0)
    right = oracle(param1, state1, input1)
    return state0 + 0.5 * step * (left + right)


def create_integrator(
    integrator: Callable,
    oracle: OracleProtocol,
    num_states: int,
    num_inputs: int,
    num_params: int = 0,
) -> OracleProtocol:
    """
    Build integrator object.

    Builder function to create an integrator casadi_function. This function
    can then be used normally in simulations with real numbers or in casadi
    frameworks using symbolics.

    Parameters
    ----------
    integrator: OracleProtocol
        One of the integrator functions defined in this module
    oracle: OracleProtocol
        Function returning derivative to be integrated
    num_states: int
        Number of states model oracle expects
    num_inputs: int
        Number of inputs model oracle expects
    num_params: int = 0
        Number of parameters model oracle expects, defaults to zero for backward
        compatibility.

    Returns
    -------
    integ_obj: OracleProtocol
        Integrator object which can be used in casadi framework or normally

    """
    if integrator is trapz:
        func, arglist = _create_implicit_integrator_no_params(
            integrator=integrator,
            oracle=oracle,
            num_states=num_states,
            num_inputs=num_inputs,
            num_params=num_params,
        )
    else:
        func, arglist = _create_explicit_integrator_no_params(
            integrator=integrator,
            oracle=oracle,
            num_states=num_states,
            num_inputs=num_inputs,
            num_params=num_params,
        )

    return (cb.casadi_function(arglist, func_name=integrator.__name__))(func)


def _create_explicit_integrator_no_params(
    integrator: Callable,
    oracle: OracleProtocol,
    num_states: int,
    num_inputs: int,
    num_params: int,
) -> tuple:
    if num_params > 0:
        part_int = ftool.partial(integrator, oracle)
        arglist = (num_params, num_states, num_inputs, 1)
        return part_int, arglist

    def new_oracle(_, states_vec, inputs_vec):
        return oracle(states_vec, inputs_vec)

    part_int = ftool.partial(integrator, new_oracle, ca.DM(0))
    arglist = (num_states, num_inputs, 1)
    return part_int, arglist


def _create_implicit_integrator_no_params(
    integrator: Callable,
    oracle: OracleProtocol,
    num_states: int,
    num_inputs: int,
    num_params: int,
) -> tuple:
    if num_params > 0:
        part_int = ftool.partial(integrator, oracle)
        arglist = (
            num_params,
            num_params,
            num_states,
            num_states,
            num_inputs,
            num_inputs,
            1,
        )
        return part_int, arglist

    def new_oracle(_, states_vec, inputs_vec):
        return oracle(states_vec, inputs_vec)

    part_int = ftool.partial(integrator, new_oracle, ca.DM(0), ca.DM(0))
    arglist = (num_states, num_states, num_inputs, num_inputs, 1)
    return part_int, arglist
