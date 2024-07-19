"""
Implment utilities to create casadi objects.

This module implements the a few utilities to facilitate the easy create of
casadi function objects from normal python objects.

"""

from dataclasses import dataclass
from inspect import Signature, signature
from typing import Callable, Iterable

import casadi as ca


def _get_param_names(sig: Signature) -> tuple[str]:
    params = [param for param in sig.parameters.keys() if not param == "self"]
    return (*params,)


def _create_symbols(params: tuple[str], argsizes: tuple[int]) -> tuple[ca.MX]:
    sym_params = []
    for param, size in zip(params, argsizes):
        sym_param = ca.MX.sym(param, size, 1)
        sym_params.append(sym_param)

    return (*sym_params,)


def _create_symbolic_func(
    func: Callable,
    name: str,
    params: tuple[str],
    sym_params: tuple[ca.MX],
    num_output: int,
) -> ca.Function:

    if num_output == 1:
        out_list = [func(*sym_params)]
        out_names = ["output"]
    else:
        out_list = [func(*sym_params)[num] for num in range(num_output)]
        out_names = [f"output_{num}" for num in range(num_output)]

    sym_method_mx = ca.Function(name, [*sym_params], out_list, [*params], out_names)
    try:
        sym_method_out = sym_method_mx.expand()
    except RuntimeError:
        sym_method_out = sym_method_mx

    return sym_method_out


def _register_casadi_methods(cls) -> dict[str, tuple[int]]:
    registered_methods = {}

    for methodname in dir(cls):
        method = getattr(cls, methodname)
        try:
            argsizes, num_out = getattr(method, "_casadi")
        except AttributeError:
            continue

        registered_methods[methodname] = (argsizes, num_out)

    return registered_methods


def _post_init(self) -> None:
    for name, (argsizes, num_out) in self._casadi_methods.items():
        method = getattr(self, name)
        sig = signature(method)

        params = _get_param_names(sig)

        symbols = _create_symbols(params, argsizes)

        sym_method = _create_symbolic_func(method, name, params, symbols, num_out)
        object.__setattr__(self, name, sym_method)


def casadi_method(argsizes: Iterable[int], *, num_outputs: int = 1) -> Callable:
    """
    Convert method into casadi_method.

    This decorator takes a plain method and converts it into a casadi functor
    object. It then replaces the plain method with the newly created casadi
    functor. This function must be used in tandem with a casadi_dataclass
    because methods must be converted after they are bound to an instance.


    Parameters
    ----------
    argsizes: Iterable[int]
        Iterable of the length of each argument vector to the function
    num_outputs: int
        Number of output objects

    Returns
    -------
    casadi_method: Callable
        Casadi functor object

    """

    def _casadi_method_marker(method):
        method._casadi = (argsizes, num_outputs)
        return method

    return _casadi_method_marker


def casadi_function(
    argsizes: Iterable[int], *, num_outputs: int = 1, func_name: str = None
) -> Callable:
    """
    Convert function into casadi_function.

    This decorator takes a plain function and converts it into a
    casadi functor object. It then replaces the plain function with the newly
    created casadi functor.

    Parameters
    ----------
    argsizes: Iterable[int]
        Iterable of the length of each argument vector to the function
    num_outputs: int
        Number of output objects
    func_name: str = None
        Optional function name override

    Returns
    -------
    casadi_func: Callable
        Casadi functor object

    """

    def _casadi_func_generator(func: Callable):
        sig = signature(func)
        params = _get_param_names(sig)

        if not func_name:
            name = func.__name__
        else:
            name = func_name

        syms = _create_symbols(params, argsizes)
        sym_func = _create_symbolic_func(func, name, params, syms, num_outputs)

        return sym_func

    return _casadi_func_generator


def casadi_dataclass(cls):
    """
    Convert class into casadi_dataclass.

    This decorator must be used on any class implementing casadi_methods.
    It automatically turns the class into a dataclass which is useful for
    the most common use case of creating dynamic system models. The class
    is frozen so that parameters cannot be changed after instantiation. Because
    casadi_methods are created at instantiation, any changes to parameters
    after instantiation will not be reflected in the casadi_methods. This
    means parameters cannot be set in the __post_init__ method. As a workaround,
    a user can make these fields into properties rather than member variables.

    Note
    -------
    casadi_dataclasses are frozen meaning no member variables can be changed
    after instantiation. As a result __post_init__ methods are not supported.
    Consider making fields into properties as a workaround.

    Parameters
    ----------
    cls:
        Python class instance

    Returns
    -------
    casadi_cls: dataclass
       dataclass with casadi methods

    """
    cls._casadi_methods = _register_casadi_methods(cls)

    def noop(*args, **kwargs):
        pass

    try:
        old_init = cls.__post_init__
    except AttributeError:
        old_init = noop

    def wrap_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        _post_init(self)

    cls.__post_init__ = wrap_init
    cls = dataclass(cls, frozen=True)

    return cls
