import abc
import functools
import pickle
from collections import deque
from dataclasses import InitVar, dataclass, field
from numbers import Number
from typing import IO, Iterable, NamedTuple

import numpy as np

from casadi_tools.dynamics import named_arrays as na
from casadi_tools.nlp_utils import nlp_elems, nlp_runner


@dataclass
class BaseLogger(abc.ABC):
    """Log simulation results in deque member variables."""

    state_names: Iterable[str]
    input_names: Iterable[str]

    states: dict[str, deque] = field(init=False, repr=False, default_factory=dict)
    """Dict of data names to deques of values over time"""
    inputs: dict[str, deque] = field(init=False, repr=False, default_factory=dict)
    """Dict of data names to deques of values over time"""
    times: dict[str, deque] = field(init=False, repr=False, default_factory=dict)
    """Deque of simulation epochs."""

    @abc.abstractmethod
    def as_dicts(self) -> tuple[dict[str, np.ndarray] | np.ndarray]:
        """Convert logger object into pickleable objects"""

    @staticmethod
    def _field_to_dict(field: dict[str, deque]) -> dict[str, np.ndarray]:
        return {key: pack_to_numpy_array(value) for key, value in field.items()}

    def log_states(self, states: na.NamedBase | dict[str, np.ndarray]):
        """
        Log current states.

        Parameters
        ----------
        states: NamedBase
            Current system states

        """
        for key in self.state_names:
            self.states[key].append(states[key])

    def log_inputs(self, inputs: na.NamedBase | dict[str, np.ndarray]):
        """
        Log current inputs.

        Parameters
        ----------
        inputs: NamedBase
            Current system inputs

        """
        for key in self.input_names:
            self.inputs[key].append(inputs[key])

    def pickle(self, pickle_file: IO[bytes]) -> None:
        """
        Pickle simulation logger.

        This method pickles the logger class as a dictionary
        of state, input, and time values. See `SimLogger.convert_to_dictionary`
        for more information

        Parameters
        ----------
        pickle_file: file object
            File object to dump pickle into

        """
        pickle.dump(self.as_dicts(), pickle_file)

    def __post_init__(self):
        for name in self.state_names:
            self.states[name] = deque()

        for name in self.input_names:
            self.inputs[name] = deque()


@dataclass
class SimLogger(BaseLogger):
    """
    Simulation logger for logging sim results.

    Simple logging class for a dynamic simulation. It holds states, inputs, and times
    in deque objects.

    """

    other_names: Iterable[str] = field(default_factory=list)

    others: dict[str, deque] = field(init=False, repr=False, default_factory=dict)
    """Dict of data names to deques of values over time"""

    def log_data(
        self,
        states: na.NamedBase,
        inputs: na.NamedBase,
        time: float,
        others: na.NamedBase | dict[str, np.ndarray] = None,
    ):
        """Log all current data at one time."""

        self.log_states(states)
        self.log_inputs(inputs)
        self.log_time(time)

        if others is not None:
            self.log_others(others)

    def log_others(self, other: na.NamedBase | dict[str, np.ndarray]):
        """
        Log current other variables.

        Parameters
        ----------
        other: NamedBase | dict[str, np.ndarray]
            Current other variables

        """
        for key in self.other_names:
            self.others[key].append(other[key])

    def right_pad_inputs(self, pad_val: Number = np.NaN):
        """
        Pad inputs with NaN's on right side.

        Allows easy plotting by padding input arrays to be the same size as the
        time array. Input arrays will naturally be one element shorter than time
        or state arrays due to forward projection.

        Parameters
        ----------
        pad_val: Number = np.NaN
            Value to pad with

        """
        for key in self.input_names:
            self.inputs[key].append(pad_val)

    def log_time(self, epoch: float) -> None:
        """
        Log current time.

        Parameters
        ----------
        time: float
            Current simulation epoch

        """
        self.times["time"].append(epoch)

    def as_dicts(self) -> tuple[dict[str, np.ndarray] | np.ndarray]:
        state_out = self._field_to_dict(self.states)
        input_out = self._field_to_dict(self.inputs)
        others_out = self._field_to_dict(self.others)
        time_out = pack_to_numpy_array(self.times)
        return time_out, state_out, input_out, others_out

    def __post_init__(self):
        super().__post_init__()
        self.times["time"] = deque()
        for name in self.other_names:
            self.others[name] = deque()


class NLPPickledData(NamedTuple):
    time: dict[str, np.ndarray]
    states: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray]
    other: dict[str, np.ndarray]
    params: dict[str, np.ndarray]
    stats: dict[str, np.ndarray]


@dataclass
class NLPLogger(BaseLogger):
    """
    NLP logger for logging nlp sim results.

    Simple logging class for an nlp simulation. It holds states and inputs
    in deque objects.

    """

    other_names: Iterable[str]
    param_names: Iterable[str]

    other: dict[str, deque] = field(init=False, repr=False, default_factory=dict)
    """Dict of data names to deques of values over time"""
    params: dict[str, deque] = field(init=False, repr=False, default_factory=dict)
    """Dict of data names to deques of values over time"""
    stats: dict[str, deque] = field(init=False, repr=False, default_factory=dict)
    """Dict of data names to deques of values over time"""

    num_horizons: InitVar[int]
    """ Number of nlp multistage problems"""

    def log_data(
        self,
        *,
        data: dict[str, np.ndarray],
        params: dict[str, np.ndarray],
        time: np.ndarray,
        stats: nlp_runner.NLPSolveStats,
    ):
        """Log all current data at one time."""
        self.log_states(data)
        self.log_inputs(data)
        self.log_other(data)
        self.log_params(params)
        self.log_time(time)
        self.log_stats(stats)

    def log_params(self, params: nlp_elems.ParameterVariable):
        """
        Log current parameters.

        Parameters
        ----------
        params: dict[str, np.ndarray]
            Current parameters

        """
        for key in self.param_names:
            self.params[key].append(params.get_value(key))

    def log_other(self, other: na.NamedBase | dict[str, np.ndarray]):
        """
        Log current other variables.

        Parameters
        ----------
        other: NamedBase | dict[str, np.ndarray]
            Current other variables

        """
        for key in self.other_names:
            self.other[key].append(other[key])

    @functools.singledispatchmethod
    def log_time(self, horizon) -> None:
        raise NotImplementedError(
            "horizon must be a numpy array or dict of numpy arrays."
        )

    @log_time.register(np.ndarray)
    def _(self, horizon: np.ndarray) -> None:
        """
        Log the current horizon.

        Parameters
        ----------
        horizon: dict[str, np.ndarray]
            Horizon numpy array

        """
        self.times["time__0"].append(horizon)

    @log_time.register(dict)
    def _(self, horizon: dict[str, np.ndarray]) -> None:
        """
        Log the current set of horizons.

        Parameters
        ----------
        horizon: dict[str, np.ndarray]
            Dictionary of each multistage horizon and corresponding numpy array

        """
        for key, val in self.times.items():
            val.append(horizon[key])

    def log_stats(self, stats: nlp_runner.NLPSolveStats) -> None:
        """
        Log solver stats.

        Parameters
        ----------
        exit_flag: int
            Exit flag of the solver
        iter: int
            Number of solver iterations
        solve_time: float
            Elapsed solve time [ms]
        cost: float
            Total cost incurred during current solve

        """
        for key, value in self.stats.items():
            value.append(getattr(stats, key))

    def __post_init__(self, num_horizons):
        super().__post_init__()

        for idx in range(num_horizons):
            self.times[f"time__{idx}"] = deque()

        for key in self.param_names:
            self.params[key] = deque()

        for key in self.other_names:
            self.other[key] = deque()

        for key in nlp_runner.NLPSolveStats._fields:
            self.stats[key] = deque()

    def as_dicts(self) -> NLPPickledData:
        state_out = self._field_to_dict(self.states)
        state_out.update(self.merge_results(state_out))

        input_out = self._field_to_dict(self.inputs)
        input_out.update(self.merge_results(input_out))

        other_out = self._field_to_dict(self.other)
        other_out.update(self.merge_results(other_out))

        params_out = self._field_to_dict(self.params)
        params_out.update(self.merge_results(params_out))

        time_out = self._field_to_dict(self.times)
        time_out.update(self.merge_results(time_out))

        stats_out = self._field_to_dict(self.stats)

        return NLPPickledData(
            time=time_out,
            states=state_out,
            inputs=input_out,
            params=params_out,
            other=other_out,
            stats=stats_out,
        )

    @staticmethod
    def merge_results(results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        temp_dict = {}
        for key, val in results.items():
            base_name, *_ = key.rsplit(sep="__", maxsplit=1)
            try:
                temp_dict[base_name].append(val)
            except KeyError:
                temp_dict[base_name] = [val]

        return {key: np.concatenate(val, axis=1) for key, val in temp_dict.items()}

    @staticmethod
    def extract_single_stage(
        results: tuple[dict[str, np.ndarray]], stage: int
    ) -> tuple[dict[str, np.ndarray]]:
        elem_list = []
        for elem in results:
            value_dict = {}
            for key, val in elem.items():
                try:
                    name, idx = key.rsplit(sep="__", maxsplit=1)
                    if int(idx) == stage:
                        value_dict[name] = val
                except ValueError:
                    continue

            elem_list.append(value_dict)
        return elem_list


@functools.singledispatch
def pack_to_numpy_array(in_var) -> None:
    """
    Pack to numpy array.

    Single dispatch method can be called with either a dictionary, an
    iterable, a numpy array, or a scalar as the argument

    Called with a dictionary, this function takes the iterable dictionary
    values and converts those iterables to numpy arrays.

    Called with an iterable, the iterable is converted to a numpy array.

    In either case, if the iterable contains numpy arrays itself,
    those arrays are stacked row-wise to yield a 2D matrix with initially
    logged values on top and the last logged values on the bottom.

    Called with a numpy array, it squeezes the input and returns the
    squeezed array.

    Called with a scalar, it simply returns the input number

    Parameters
    ----------
    in_var:
        Object to convert

    Returns
    -------
    out_var:
        Object containing numpy arrays rather than iterables

    """
    raise NotImplementedError


@pack_to_numpy_array.register(float)
@pack_to_numpy_array.register(int)
def _(in_var) -> Number:
    return in_var


@pack_to_numpy_array.register
def _(in_var: np.ndarray) -> np.ndarray:
    return in_var.squeeze()


@pack_to_numpy_array.register(deque)
@pack_to_numpy_array.register(list)
@pack_to_numpy_array.register(tuple)
def _(in_var) -> np.ndarray:
    return np.row_stack(tuple(in_var))


@pack_to_numpy_array.register
def _(in_var: dict) -> dict[str, Iterable[np.ndarray]]:
    out_dict = {}

    for key, val in in_var.items():
        out_dict[key] = np.row_stack(tuple(val))

    return out_dict
