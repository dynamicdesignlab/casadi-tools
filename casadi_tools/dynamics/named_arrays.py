"""
Implement NamedVector class and utilities.

This module implements the named vector class. It is designed to provide a
friendly data structure for use in dynamic modeling and interactions with
casadi symbolics

"""

import abc
import dataclasses
from dataclasses import dataclass, field
from numbers import Number, Real
from typing import Any, ClassVar, Generator, Iterable, Type, TypeVar
from typing_extensions import Self

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate

from casadi_tools import types

FIELD_TYPE = types.CASADI_TYPE | float | np.ndarray
NV_FIELD_TYPE = float | types.CASADI_SYMBOLIC
NA_FIELD_TYPE = np.ndarray | types.CASADI_SYMBOLIC

T = TypeVar("T", bound="NamedBase")
TA = TypeVar("TA", bound="NamedArray")
TV = TypeVar("TV", bound="NamedVector")


@dataclass
class NamedBase(abc.ABC):
    """
    Base array-like object which can be accessed with dot notation.

    This class implements the base functionality for Named-ArrayLike objects
    including math methods and builtins.

    """

    @classmethod
    @property
    def num_fields(cls) -> int:
        """Get number of fields in named vector."""
        return len(cls.field_names)

    @classmethod
    @property
    @abc.abstractmethod
    def field_names(cls) -> tuple[str]:
        """Get names of fields in named vector."""

    def __iter__(self) -> Generator[Any, None, None]:
        return self.values()

    def __neg__(self) -> Self:
        return self.__class__(*(-elem for elem in self))

    def __add__(self, other: Self) -> Self:
        return self.__class__(*(ours + theirs for ours, theirs in zip(self, other)))

    def __sub__(self, other: Self) -> Self:
        return self.__class__(*(ours - theirs for ours, theirs in zip(self, other)))

    def __mul__(self, scalar: Real) -> Self:
        return self.__class__(*(elem * scalar for elem in self))

    def __rmul__(self, scalar: Real) -> Self:
        return self * scalar

    def __truediv__(self, scalar: Real) -> Self:
        return self.__class__(*(elem / scalar for elem in self))

    def __getitem__(self, name: str) -> FIELD_TYPE:
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(f"No field named {name}")

    def get(self, name) -> FIELD_TYPE:
        """Alias for __getitem__."""
        return self.__getitem__(name)

    def values(self) -> Generator[FIELD_TYPE, None, None]:
        """
        Return generator of field values.

        If field values are casadi symbolics, then those symbolics are returned,
        otherwise, floats are returned.

        Yields
        ------
        value: FIELD_TYPE
            field value

        """
        return (self[name] for name in self.field_names)

    def items(self) -> Generator[tuple[str, FIELD_TYPE], None, None]:
        """
        Return generator of field names and their corresponding values.

        If field values are casadi symbolics, then those symbolics are returned,
        otherwise, floats are returned.

        Yields
        ------
        name: str
            field name
        value: FIELD_TYPE
            field value

        """
        return zip(self.field_names, self.values())

    def asdict(self) -> dict[str, Any]:
        """
        Convert namedvector to dictionary.

        Returns
        -------
        dict[str, Any]:
            Dictionary equivalent of namedvector

        """
        return {key: value for key, value in self.items()}

    @abc.abstractmethod
    def to_casadi_array(self) -> types.CASADI_TYPE:
        """
        Convert named array-like to casadi array.

        This method returns a casadi array of the named array-likes's
        field values.

        Returns
        -------
        array: cb.CASADI_TYPE
            Casadi array of named vector's field values

        """

    @abc.abstractmethod
    def to_numpy_array(self) -> np.ndarray:
        """
        Convert named array-like to casadi array.

        This method returns a casadi array of the named array-likes's
        field values.

        Returns
        -------
        array: np.ndarray
            Numpy array of named vector's field values

        """

    @abc.abstractclassmethod
    def from_array(cls: type[T], array: ArrayLike | types.CASADI_TYPE) -> T:
        """
        Convert array to named array-like.

        This method returns a named vector from an array-like object. The fields
        are filled in the order they were declared.

        Parameters
        ----------
        array: ArrayLike | types.CASADI_TYPE
            Iterable filled with values to fill named vector's fields

        Returns
        -------
        NamedBase

        """

    def to_array(self) -> Any:
        """Convert to array-like object, preserving internal datatype."""
        if isinstance(next(self.values()), types.CASADI_SYMBOLIC):
            return self.to_casadi_array()
        else:
            return self.to_numpy_array()

    @property
    def field_type(self) -> Type:
        """Return the underlying field type."""
        return type(next(self.values()))


@dataclass
class NamedVector(NamedBase):
    """
    Vector which can be accessed with dot notation.

    NamedVector objects function as an interface
    between casadi functions which take only normal vectors and user functions
    which for simplicity and error reduction should take classes with named
    attributes. Casadi symbolic types are preserved internally, but all other types
    are cast as floats.

    Users should prefer the ``NamedVector.create_from_field_names`` class method
    to define new derived classes.

    Notes
    -----
    Casts all non-symbolic types as floats (including Casadi DM)

    Warning
    -------
    Only data fields can be added in child classes. Undefined behavior will result
    from non-data fields being added.

    """

    @classmethod
    @property
    def field_names(cls: type[TV]) -> tuple[str]:
        return tuple([item.name for item in dataclasses.fields(cls)])

    def approx_eq(self: TV, other: TV) -> bool:
        """
        Check that all field arrays are approximately equal.

        Only NamedArrays with numpy field type can be compared with this method.

        Parameters
        ----------
        other: NamedBase
            Other NamedBase type to check approximate equality

        Returns
        -------
        bool
            True if approximately equal

        Raises
        ------
        TypeError
            If types are not the same or field types are not numpy arrays or casadi DM

        """
        if not type(self) is type(other):
            raise TypeError("Cannot compare incompatible types")
        elif not issubclass(self.field_type, Number | ca.DM):
            raise TypeError("Cannot approximately compare symbolic types")

        return np.allclose(self.to_numpy_array(), other.to_numpy_array())

    def to_numpy_array(self: TV) -> np.ndarray:
        """
        Convert named vector to numpy array.

        This method returns a 1-D numpy array of the named vector's
        field values. The values are ordered the same as the fields are
        declared. All field values are cast as floats.

        Returns
        -------
        vector: np.ndarray
            1-D numpy array of the named vector's field values

        Notes
        -----
        Casadi and numpy pack their arrays in the opposite fashion, with
        numpy using the C convention of row-wise, and Casadi using the Fortran
        convention of column-wise.

        """
        return np.fromiter(self.values(), dtype=float)

    def to_casadi_array(self: TV) -> types.CASADI_TYPE:
        """
        Convert named vector to casadi array.

        This method returns a casadi array of the named vector's
        field values. The values are ordered the same as the fields are
        declared.

        Returns
        -------
        vector: cb.CASADI_TYPE
            1-D casadi array of the named vector's field values

        Notes
        -----
        Casadi and numpy pack their arrays in the opposite fashion, with
        numpy using the C convention of row-wise, and Casadi using the Fortran
        convention of column-wise. This function outputs a casadi array so it will
        be a column vector.

        """
        return ca.vertcat(*self.values())

    @classmethod
    def from_array(cls: type[TV], array: np.ndarray | types.CASADI_TYPE) -> TV:
        """
        Convert array-like object to named vector.

        This method returns a named vector from an iterable. The fields are
        filled in the order they were declared. Array must have no more than one
        dimension with size greater than 1. Each field will effectively be
        scalars.

        Parameters
        ----------
        array: ArrayLike | types.CASADI_TYPE
            ArrayLike filled with values to fill named vector's fields
        floats: bool, default=False
            Option to convert all fields to floats before returning

        Returns
        -------
        named_vector: NamedVector
            Named vector object whose field values match the vector's values.

        Notes
        -----
        Array must have no more than one dimension with size greater than 1.

        """
        try:
            cls_args = array.tolist()
        except AttributeError:
            cls_args = ca.vertsplit(array)

        try:
            return cls(*cls_args)
        except TypeError as exc:
            raise ValueError("Input array has wrong shape") from exc

    @classmethod
    def new_with_validation(cls: type[TV], *args, **kwargs) -> TV:
        """
        Create a new instance of the class after validating all field values.

        Raises
        ------
        ValueError
            If field is not a single element

        """
        first_arg = next(iter(kwargs.values()))

        if isinstance(first_arg, types.CASADI_TYPE):
            field_args = cls._validate_and_create_symbolic(cls, *args, **kwargs)
        else:
            field_args = cls._validate_and_create_real(cls, *args, **kwargs)

        return cls(**field_args)

    @classmethod
    def _validate_and_create_symbolic(
        cls: type[TV], *args, **kwargs: dict[str, types.CASADI_TYPE]
    ) -> dict[str, Any]:
        field_args = {}
        for name, value in kwargs.items():
            if value.is_scalar():
                field_args[name] = value
            else:
                raise ValueError(f"Invalid size for field {name}")
        return field_args

    @classmethod
    def _validate_and_create_real(cls: type[TV], *args, **kwargs) -> dict[str, Any]:
        field_args = {}
        for name, value in kwargs.items():
            try:
                field_args[name] = float(value)
            except TypeError as exc:
                raise ValueError(f"Invalid type or size field {name}") from exc
            except RuntimeError as exc:
                raise ValueError(f"Invalid DM size for field {name}") from exc

        return field_args

    @classmethod
    def create_from_field_names(
        cls: type[T], cls_name: str, field_names: Iterable[str]
    ) -> type[T]:
        """
        Create new derived class.

        Parameters
        ----------
        cls_name: str
            Name of new class
        field_names: Iterable[str]
            Iterable of field names

        Returns
        -------
        derived_class

        """
        field_list = [(name, NV_FIELD_TYPE) for name in field_names]
        return dataclasses.make_dataclass(
            cls_name=cls_name, fields=field_list, bases=(cls,), eq=False
        )


@dataclass
class NamedArray(NamedBase):
    """
    Array which can be accessed with dot notation.

    NamedArray objects function as an interface between casadi functions which
    take only normal arrays and user functions which for simplicity and error
    reduction should take classes with named attributes. Casadi symbolic types
    are preserved internally, but all other types are cast as numpy arrays of
    floats. Each field represents a 1-D numpy array.

    Users should prefer the ``NamedArray.create_from_field_names`` class method
    to define new derived classes.

    Notes
    -----
    Casts all non-symbolic types as numpy arrays of floats (including Casadi DM).

    Warning
    -------
    Only data fields can be added in child classes. Undefined behavior will result
    from non-data fields being added.

    Warning
    -------
    All inheriting classes must be dataclasses with the option ``eq=False``. If
    this option is not set, checking equality among NamedArrays is undefined
    behavior.

    """

    field_length: int = field(init=False)
    _ignored_fields: ClassVar[set[str]] = {"field_length"}

    def __post_init__(self: TA):
        first_field = next(iter(self.values()))
        try:
            self.field_length = first_field.size1()
        except AttributeError:
            self.field_length = len(first_field)

    @classmethod
    @property
    def field_names(cls: type[TA]) -> tuple[str]:
        """Get names of fields in named vector."""
        return tuple(
            [
                item.name
                for item in dataclasses.fields(cls)
                if item.name not in cls._ignored_fields
            ]
        )

    def __eq__(self: TA, other: Self) -> bool:
        """
        Check that all field arrays are equal.

        Only NamedArrays with numpy or SX field type can be compared with this method.

        Parameters
        ----------
        other: NamedArray
            Other array to check equality

        Returns
        -------
        bool
            True if equal

        Raises
        ------
        TypeError
            If types are not the same or field types are not np arrays or SX symbolics.

        """
        if not type(self) is type(other) or not self.field_type == other.field_type:
            raise TypeError("Cannot compare incompatible types")

        if self.field_type == ca.MX:
            raise TypeError("Cannot compare MX symbolic types")
        elif self.field_type == ca.SX:
            return ca.is_equal(self.to_casadi_array(), other.to_casadi_array())
        else:
            return np.array_equiv(self.to_numpy_array(), other.to_numpy_array())

    def approx_eq(self: TA, other: TA) -> bool:
        """
        Check that all field arrays are approximately equal.

        Only NamedArrays with numpy field type can be compared with this method.

        Parameters
        ----------
        other: NamedArray
            Other array to check approximate equality

        Returns
        -------
        bool
            True if approximately equal

        Raises
        ------
        TypeError
            If types are not the same or field types are not numpy arrays

        """
        if not type(self) is type(other) or not self.field_type == other.field_type:
            raise TypeError("Cannot compare incompatible types")
        elif not self.field_type == np.ndarray:
            raise TypeError("Cannot approximately compare symbolic types")

        return np.allclose(self.to_numpy_array(), other.to_numpy_array())

    @classmethod
    def from_array(
        cls: type[TA], array: ArrayLike | types.CASADI_TYPE, axis: int = None
    ) -> T:
        """
        Convert array-like object to named array.

        This method returns a named vector from an iterable. The fields are
        filled in the order they were declared. Array must be 2-D and have at
        least one dimension that is equal to the number of dataclass fields. If
        no axis is specified and there is exactly one dimension equal to the
        number of fields, the input array is split along this dimension, and the
        fields are filled with the resulting 1-D arrays. If both dimensions are
        equal to the number of fields, and axis must be specified.

        Parameters
        ----------
        array: ArrayLike | types.CASADI_TYPE
            ArrayLike filled with values to fill named arrays' fields
        axis: int = None
            Dimension along which to split input array

        Returns
        -------
        named_array: NamedArray
            Named array object whose field values match the array's values.

        Notes
        -----
        Array must be 2-D and have at least one dimension equal to the number of fields.

        Raises
        ------
        ValueError
            If input array is not appropriately sized or square matrix is given
            with no axis argument.

        """
        if not isinstance(array, types.CASADI_SYMBOLIC):
            args = cls._from_real_array(array=array, axis=axis)
        else:
            args = cls._from_symbolic_array(array=array, axis=axis)

        return cls(*args)

    @classmethod
    def _from_symbolic_array(
        cls: type[TA], array: ArrayLike | types.CASADI_TYPE, axis: int
    ) -> TA:
        if array.shape[0] == array.shape[1] and axis is None:
            raise ValueError(
                '"axis" argument must be used to disambiguate square matrix'
            )
        elif axis == 0 and array.size1() == cls.num_fields:
            return ca.horzsplit(array.T)
        elif axis == 1 and array.size2() == cls.num_fields:
            return ca.horzsplit(array)
        elif array.size1() == cls.num_fields:
            return ca.horzsplit(array.T)
        elif array.size2() == cls.num_fields:
            return ca.horzsplit(array)
        else:
            raise ValueError("Input array has the wrong shape")

    @classmethod
    def _from_real_array(
        cls: type[TA], array: ArrayLike | types.CASADI_TYPE, axis: int
    ) -> TA:
        array = np.array(array)

        if array.shape[0] == array.shape[1] and axis is None:
            raise ValueError(
                '"axis" argument must be used to disambiguate square matrix'
            )
        elif axis == 0 and array.shape[0] == cls.num_fields:
            args = np.vsplit(array, cls.num_fields)
        elif axis == 1 and array.shape[1] == cls.num_fields:
            args = np.hsplit(array, cls.num_fields)
        elif array.shape[0] == cls.num_fields:
            args = np.vsplit(array, cls.num_fields)
        elif array.shape[1] == cls.num_fields:
            args = np.hsplit(array, cls.num_fields)
        else:
            raise ValueError("Input array has the wrong shape")

        return tuple(arr.squeeze() for arr in args)

    @classmethod
    def from_namedarray(cls: type[TA], name_array: TA) -> TA:
        """
        Create new named array from another named array.

        This method returns a named array instance by extracting and copying the
        fields the two arrays have in common. The named array to be created must
        be a subset of the name_array parameter.

        Parameters
        ----------
        name_array: NamedArray
            NamedArray to extract field values from

        Returns
        -------
        NamedArray
            Named array instance with copied field values

        Raises
        ------
        AttributeError
            If name_array parameter does not have all necessary fields.

        """
        args = [getattr(name_array, name) for name in cls.field_names]
        return cls(*args)

    def to_casadi_array(self: TA) -> types.CASADI_TYPE:
        """
        Output field values as a casadi array.

        Field value arrays are stacked row-wise so that each row in the output
        array represents a field, and each column represents a stage.

        Returns
        -------
        out_array: types.CASADI_TYPE
            2-D array with fields as rows

        """
        return ca.horzcat(*self.values()).T

    def to_numpy_array(self: TA) -> np.ndarray:
        """
        Output field values as a numpy array.

        Field value arrays are stacked row-wise so that each row in the output
        array represents a field, and each column represents a stage.

        Returns
        -------
        out_array: np.ndarray:
            2-D numpy array with fields as rows

        """
        return np.row_stack(tuple(self.values()))

    def values(self: TA) -> Generator[NA_FIELD_TYPE, None, None]:
        """
        Return generator of field values.

        If field values are casadi symbolics, then those symbolics are returned,
        otherwise, numpy arrays of floats are returned.

        Yields
        ------
        value: NA_FIELD_TYPE
            field value

        """
        return (getattr(self, name) for name in self.field_names)

    def items(self: TA) -> Generator[tuple[str, NA_FIELD_TYPE], None, None]:
        """
        Return generator of field names and their corresponding values.

        If field values are casadi symbolics, then those symbolics are returned,
        otherwise, numpy arrays of floats are returned.

        Yields
        ------
        name: str
            field name
        value: NA_FIELD_TYPE
            field value

        """
        return zip(self.field_names, self.values())

    def array_at_index(self: TA, idx: slice) -> NA_FIELD_TYPE:
        """
        Return slice across all fields at array index.

        Parameters
        ----------
        idx: slice
            Slice object

        Returns
        -------
        NA_FIELD_TYPE:
            Slice of field arrays

        """
        return self.to_array()[:, idx]

    def iter_over_array(self: TA) -> Generator[NA_FIELD_TYPE, None, None]:
        """
        Return iterator of slices across all field arrays, element by element.

        Yields
        ------
        NA_FIELD_TYPE:
            Slice of field arrays

        """
        return (self.array_at_index(idx) for idx in range(self.field_length))

    def interp1d(self: TA, x: ArrayLike, xp: ArrayLike) -> ArrayLike:
        """
        Interpolate over field arrays, returning a slice at interpolated point.

        Only real types can be interpolated. Query points lying outside the given
        x reference points will return NaN.

        Parameters
        ----------
        x: ArrayLike
            Query points
        xp: ArrayLike
            X coordinate of interpolation table
        namedvector: bool = False
            If true return an associated namedvector type, otherwise a numpy array

        Returns
        -------
        y: ArrayLike
            Interpolated y-values at query points

        Raises
        ------
        TypeError
            If NamedArray is symbolic

        """
        array = self.to_array()

        if isinstance(array, types.CASADI_SYMBOLIC):
            raise TypeError("Cannot interpolate symbolic NamedArray")

        interp_array = interpolate.interp1d(
            x=xp, y=array, bounds_error=False, fill_value=(array[:, 0], array[:, -1])
        )

        return interp_array(x=x)

    @classmethod
    def create_from_namedvector(
        cls: type[TA], cls_name: str, namedvector: Type[NamedVector]
    ) -> type[TA]:
        """
        Create a class derived from NamedArray associated with a NamedVector.

        Creating a NamedArray using this method will have identical fields with
        identical ordering to the given NamedVector.

        Parameters
        ----------
        cls_name: str
            Name of derived class
        namedvector: Type[NamedVector]
            Associated NamedVector type

        Returns
        -------
        derived_class

        Raises
        ------
        TypeError
            If namedvector is not a subclass of NamedVector

        """
        return cls.create_from_field_names(
            cls_name=cls_name, field_names=namedvector.field_names
        )

    @classmethod
    def new_with_validation(cls: type[TA], *args, **kwargs) -> TA:
        """
        Create a new instance of the class after validating all field values.

        Raises
        ------
        ValueError
            If field is not a single element

        """
        first_arg = next(iter(kwargs.values()))

        if isinstance(first_arg, types.CASADI_TYPE):
            field_len = first_arg.size1()
            field_args = cls._validate_and_create_symbolic(
                cls, field_len=field_len, *args, **kwargs
            )
        else:
            field_len = np.array(first_arg).shape[0]
            field_args = cls._validate_and_create_real(
                cls, field_len=field_len, *args, **kwargs
            )

        return cls(**field_args)

    def _validate_and_create_symbolic(
        cls: type[TA], field_len: int, *args, **kwargs: dict[str, types.CASADI_SYMBOLIC]
    ) -> dict[str, types.CASADI_SYMBOLIC]:
        field_dict = {}

        for name, value in kwargs.items():
            if not value.size1() == field_len:
                raise ValueError(f"Input array for field {name} wrong size")

            field_dict[name] = value

        return field_dict

    def _validate_and_create_real(
        cls: type[TA], field_len: int, *args, **kwargs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        field_dict = {}

        for name, value in kwargs.items():
            np_val = np.array(value)
            if not np_val.shape == (field_len,):
                raise ValueError(f"Input array for field {name} wrong size")

            field_dict[name] = np_val

        return field_dict

    @classmethod
    def create_from_field_names(
        cls: type[T], cls_name: str, field_names: Iterable[str]
    ) -> type[T]:
        """
        Create new derived class.

        Parameters
        ----------
        cls_name: str
            Name of new class
        field_names: Iterable[str]
            Iterable of field names

        Returns
        -------
        derived_class

        """
        field_list = [(name, NA_FIELD_TYPE) for name in field_names]
        return dataclasses.make_dataclass(
            cls_name=cls_name, fields=field_list, bases=(cls,), eq=False
        )
