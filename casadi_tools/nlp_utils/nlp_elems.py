"""Implements a builder class for nlp's in casadi."""

import functools
from dataclasses import dataclass, field
from numbers import Real
from typing import Iterable, List, Tuple

import casadi as ca
import numpy as np
from typing_extensions import Self

from casadi_tools import types

IN_VAL = Real | np.ndarray


class BadArraySize(Exception):
    pass


class BadArrayShape(Exception):
    pass


@dataclass
class ParameterVariable:
    _names: List[str] = field(init=False, repr=False, default_factory=list)
    _exprs: List[types.CASADI_TYPE] = field(
        init=False, repr=False, default_factory=list
    )
    _sizes: List[int] = field(init=False, repr=False, default_factory=list)
    _values: List[np.ndarray] = field(init=False, repr=False, default_factory=list)

    def add_field(self, name: str, size: int, *, value: IN_VAL = np.NaN) -> None:
        expr = ca.MX.sym(name, size, 1)
        self._add_field(name=name, expr=expr, value=value)

    def _add_field(
        self, name: str, expr: types.CASADI_TYPE, *, value: IN_VAL = np.NaN
    ) -> None:
        if name in self._names:
            raise ValueError("Name already used.")

        size = expr.size1()
        self._names.append(name)
        self._exprs.append(expr)
        self._sizes.append(size)
        self._values.append(np.empty((size,)))

        self.set_value(name, value)

    def set_value(self, key: str, value: IN_VAL) -> None:
        idx = self._get_key_index(key)
        size = self.get_size(key)
        self._values[idx] = _gen_value_array(value, size)

    def items(self):
        return ((name, self.get_value(name)) for name in self.field_names)

    def values(self):
        return (self.get_value(name) for name in self.field_names)

    def to_array(self):
        """
        Get 2D array of casadi symbolics.

        This array is composed of row vectors representing each added variable.
        Stages are represented by columns. The order that variables were added
        is preserved in the ordering of the rows

        Returns
        -------
        dec_array: Casadi Symbolic Array
            2D array of all casadi symbols

        """
        if self.num_fields == 0:
            return None

        return ca.horzcat(*self._exprs).T

    def to_vector(self):
        """
        Get row vector of casadi symbolics.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        dec_vector: Casadi Symbolic Array
            Row vector of all casadi symbols

        """
        if self.num_fields == 0:
            return None

        return ca.vertcat(*self._exprs).T

    @property
    def value_vector(self):
        """
        Row vector of variable lower constraint value.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        lb_vector: np.ndarray
            Row vector of all values

        """
        if self.num_fields == 0:
            return None

        return np.concatenate(self._values)

    @property
    def field_names(self):
        return self._names

    @property
    def num_fields(self):
        return len(self._names)

    @property
    def num_elem(self):
        return sum((self.get_size(name) for name in self.field_names))

    def _get_key_index(self, key: str) -> int:
        return self._names.index(key)

    def get_size(self, key: str) -> int:
        idx = self._get_key_index(key)
        return self._sizes[idx]

    def get_value(self, key: str) -> np.ndarray:
        idx = self._get_key_index(key)
        return self._values[idx]

    def get_expr(self, key: str) -> types.CASADI_TYPE:
        idx = self._get_key_index(key)
        return self._exprs[idx]

    def clear_expr(self) -> None:
        self._exprs = [None for _ in range(self.num_fields)]

    def param_left_unset(self) -> bool:
        if self.num_fields == 0:
            return False

        return np.any(np.isnan(self.value_vector))

    def get_unset_parameters(self) -> Tuple[str]:
        unset = []
        for name in self.field_names:
            if np.any(np.isnan(self.get_value(name))):
                unset.append(name)

        return tuple(unset)

    @classmethod
    def merge(cls, paramvars: Iterable[Self]) -> Self:
        out_param = cls()

        for i, pvar in enumerate(paramvars):
            for name in pvar.field_names:
                out_param._add_field(
                    name=f"{name}__{i}",
                    expr=pvar.get_expr(name),
                    value=pvar.get_value(name),
                )

        return out_param

    @staticmethod
    def factory() -> Self:
        return ParameterVariable()


@dataclass
class ConstraintVariable:
    _names: List[str] = field(init=False, repr=False, default_factory=list)
    _exprs: List[types.CASADI_TYPE] = field(
        init=False, repr=False, default_factory=list
    )
    _sizes: List[int] = field(init=False, repr=False, default_factory=list)
    _lowers: List[np.ndarray] = field(init=False, repr=False, default_factory=list)
    _uppers: List[np.ndarray] = field(init=False, repr=False, default_factory=list)

    def add_field(
        self,
        name: str,
        expr: types.CASADI_TYPE,
        *,
        lower: float = -np.Inf,
        upper: float = np.Inf,
    ) -> None:
        if name in self._names:
            raise ValueError("Name already used.")

        reshaped_expr = ca.reshape(expr, -1, 1)
        size = reshaped_expr.numel()

        self._names.append(name)
        self._exprs.append(reshaped_expr)
        self._sizes.append(size)
        self._lowers.append(np.empty((size,)))
        self._uppers.append(np.empty((size,)))

        self.set_bounds(name, lower=lower, upper=upper)

    def to_array(self):
        """
        2-D array of casadi symbolics.

        This array is composed of row vectors representing each added variable.
        Stages are represented by columns. The order that variables were added
        is preserved in the ordering of the rows

        Returns
        -------
        dec_array: Casadi Symbolic Array
            2D array of all casadi symbols

        """
        return ca.horzcat(*self._exprs).T

    def to_vector(self):
        """
        Row vector of casadi symbolics.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        dec_vector: Casadi Symbolic Array
            Row vector of all casadi symbols

        """
        return ca.vertcat(*self._exprs).T

    def get_bounds(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._get_key_index(key)
        return (self._lowers[idx], self._uppers[idx])

    def set_lower_bound(self, key: str, lower: IN_VAL) -> None:
        idx = self._get_key_index(key)
        size = self.get_size(key)
        self._lowers[idx] = _gen_value_array(lower, size)

    def set_upper_bound(self, key: str, upper: IN_VAL) -> None:
        idx = self._get_key_index(key)
        size = self.get_size(key)
        self._uppers[idx] = _gen_value_array(upper, size)

    def set_bounds(self, key: str, *, lower: IN_VAL, upper: IN_VAL) -> None:
        self.set_lower_bound(key, lower)
        self.set_upper_bound(key, upper)

    @property
    def lb_vector(self):
        """
        Row vector of variable lower constraint value.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        lb_vector: np.ndarray
            Row vector of all lower box constraints

        """
        return np.concatenate(self._lowers)

    @property
    def ub_vector(self):
        """
        Row vector of variable upper constraint value.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        ub_vector: np.ndarray
            Row vector of all upper box constraints

        """
        return np.concatenate(self._uppers)

    @property
    def field_names(self):
        return self._names

    @property
    def num_fields(self):
        return len(self._names)

    @property
    def num_elem(self):
        return sum((self.get_size(name) for name in self.field_names))

    def _get_key_index(self, key: str) -> int:
        return self._names.index(key)

    def get_size(self, key: str) -> int:
        idx = self._get_key_index(key)
        return self._sizes[idx]

    def get_expr(self, key: str) -> types.CASADI_TYPE:
        idx = self._get_key_index(key)
        return self._exprs[idx]

    def clear_expr(self) -> None:
        self._exprs = [None for _ in range(self.num_fields)]

    @classmethod
    def merge(cls, cstrs: Iterable[Self]) -> Self:
        new_cstr = cls()
        for i, cstr in enumerate(cstrs):
            for name in cstr.field_names:
                lower, upper = cstr.get_bounds(name)
                new_cstr.add_field(
                    name=f"{name}__{i}",
                    expr=cstr.get_expr(name),
                    lower=lower,
                    upper=upper,
                )

        return new_cstr

    @staticmethod
    def factory() -> Self:
        return ConstraintVariable()


@dataclass
class DecisionVariable:
    _names: List[str] = field(init=False, repr=False, default_factory=list)
    _exprs: List[types.CASADI_TYPE] = field(
        init=False, repr=False, default_factory=list
    )
    _sizes: List[int] = field(init=False, repr=False, default_factory=list)
    _lowers: List[np.ndarray] = field(init=False, repr=False, default_factory=list)
    _uppers: List[np.ndarray] = field(init=False, repr=False, default_factory=list)
    _init_guesses: List[np.ndarray] = field(
        init=False, repr=False, default_factory=list
    )

    def add_field(
        self,
        name: str,
        size: int,
        lower: float = -np.Inf,
        upper: float = np.Inf,
        guess: float = 0.0,
    ) -> None:
        expr = ca.MX.sym(name, size, 1)
        self._add_field(name=name, expr=expr, lower=lower, upper=upper, guess=guess)

    def _add_field(
        self,
        name: str,
        expr: types.CASADI_TYPE,
        lower: float = -np.Inf,
        upper: float = np.Inf,
        guess: float = 0.0,
    ) -> None:
        if name in self._names:
            raise ValueError("Name already used.")

        size = expr.size1()
        self._names.append(name)
        self._sizes.append(size)
        self._exprs.append(expr)
        self._lowers.append(np.empty((size,)))
        self._uppers.append(np.empty((size,)))
        self._init_guesses.append(np.empty((size,)))

        self.set_bounds(name, lower=lower, upper=upper)
        self.set_init_guess(name, guess=guess)

    @property
    def field_names(self):
        return self._names

    @property
    def num_fields(self):
        return len(self._names)

    @property
    def num_elem(self):
        return sum(self._sizes)

    def to_array(self):
        """
        2-D array of casadi symbolics.

        This array is composed of row vectors representing each added variable.
        Stages are represented by columns. The order that variables were added
        is preserved in the ordering of the rows

        Returns
        -------
        dec_array: Casadi Symbolic Array
            2D array of all casadi symbols

        """
        return ca.horzcat(*self._exprs).T

    def to_vector(self):
        """
        Row vector of casadi symbolics.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        dec_vector: Casadi Symbolic Array
            Row vector of all casadi symbols

        """
        return ca.vertcat(*self._exprs).T

    def _get_key_index(self, key: str) -> int:
        return self._names.index(key)

    def get_expr(self, key: str) -> types.CASADI_TYPE:
        idx = self._get_key_index(key)
        return self._exprs[idx]

    def clear_expr(self) -> None:
        self._exprs = [None for _ in range(self.num_fields)]

    @property
    def lb_vector(self):
        """
        Row vector of variable lower constraint value.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        lb_vector: np.ndarray
            Row vector of all lower box constraints

        """
        try:
            return np.concatenate(self._lowers)
        except ValueError:
            return np.zeros((0,))

    @property
    def ub_vector(self):
        """
        Row vector of variable upper constraint value.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        ub_vector: np.ndarray
            Row vector of all upper box constraints

        """
        try:
            return np.concatenate(self._uppers)
        except ValueError:
            return np.zeros((0,))

    @property
    def init_guess_vector(self):
        """
        Row vector of variable upper constraint value.

        This vector preserves the order that variables were added in
        from left to right. All N elements of the first variable added will
        appear first, then all N elements of the second variable, and so on.

        Returns
        -------
        ub_vector: np.ndarray
            Row vector of all upper box constraints

        """
        try:
            return np.concatenate(self._init_guesses)
        except ValueError:
            return np.zeros((0,))

    def get_size(self, key: str) -> int:
        idx = self._get_key_index(key)
        return self._sizes[idx]

    def get_bounds(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._get_key_index(key)
        return (self._lowers[idx], self._uppers[idx])

    def get_init_guess(self, key: str) -> np.ndarray:
        idx = self._get_key_index(key)
        return self._init_guesses[idx]

    def set_lower_bound(self, key: str, lower: IN_VAL) -> None:
        idx = self._get_key_index(key)
        size = self.get_size(key)
        self._lowers[idx] = _gen_value_array(lower, size)

    def set_upper_bound(self, key: str, upper: IN_VAL) -> None:
        idx = self._get_key_index(key)
        size = self.get_size(key)
        self._uppers[idx] = _gen_value_array(upper, size)

    def set_bounds(self, key: str, *, lower: IN_VAL, upper: IN_VAL) -> None:
        self.set_lower_bound(key, lower)
        self.set_upper_bound(key, upper)

    def set_init_guess(self, key: str, guess: IN_VAL) -> None:
        idx = self._get_key_index(key)
        size = self.get_size(key)
        self._init_guesses[idx] = _gen_value_array(guess, size=size)

    @classmethod
    def merge(cls, decvars: Iterable[Self]) -> Self:
        new_dec = cls()

        for i, dec in enumerate(decvars):
            for name in dec.field_names:
                lower, upper = dec.get_bounds(name)
                new_dec._add_field(
                    name=f"{name}__{i}",
                    expr=dec.get_expr(name),
                    lower=lower,
                    upper=upper,
                    guess=dec.get_init_guess(name),
                )

        return new_dec

    @staticmethod
    def factory() -> Self:
        return DecisionVariable()


@functools.singledispatch
def _gen_value_array(value, size: int) -> np.ndarray:
    raise NotImplementedError


@_gen_value_array.register
def _(value: Real, size: int) -> np.ndarray:
    return value * np.ones((size,))


@_gen_value_array.register
def _(value: np.ndarray, size: int) -> np.ndarray:
    value = value.squeeze()
    if value.ndim > 1:
        raise BadArrayShape("Array must not have more than 1 dimension after squeeze()")

    value = value.reshape((-1,))
    if not value.shape[0] == size:
        raise BadArraySize("Array is not consistent with desired size")

    return value
