"""
Implements projection tools for use with rooster.

This module implements a few tools to project different models
and data forward in time. This is useful for delay compensation

"""

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import casadi as ca
import numpy as np
from casadi_tools.dynamics import named_arrays as na
from numpy.typing import ArrayLike

T = TypeVar("T", bound="NonlinearProjector")


def calculate_epochs(
    *, proj_time: float = None, num_epochs: int = None, step: float
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Calculate simulation epochs.

    From the total simulation time and simulation step, calculate the input
    and output epochs and number of simulation stages. The output epochs are
    offset by one step from the input epochs.

    Client must provide either proj_time or num_epochs.  If provided, the
    total simulation time must be an exact multiple of the simulation step
    size.

    Parameters
    ----------
    proj_time: float
        Total simulation time
    num_epochs: int
        Number of simulation epochs
    step: float
        Simulation step size

    Returns
    -------
    input_epochs: np.ndarray
        Epochs at which inputs are applied
    output_epochs: np.ndarray
        Epochs at which outputs occur
    num_epochs: int
        Number of epochs in simulation
    proj_time: float
        Time projected forward [s]

    Raises
    ------
    ValueError
        If proj_time is not a multiple of step

    """
    if proj_time is not None and num_epochs is not None:
        raise ValueError("Cannot specify both proj_time and num_epochs")
    elif num_epochs is not None:
        proj_time = num_epochs * step
    elif proj_time is not None:
        num_epochs_float = proj_time / step
        num_epochs = int(num_epochs_float)

        if num_epochs != num_epochs_float:
            raise ValueError("proj_time is not a multiple of step")
    else:
        raise ValueError("Must specify either proj_time or num_epochs")

    input_epochs = np.linspace(
        start=0.0, stop=proj_time, num=num_epochs, endpoint=False
    )
    output_epochs = np.append(input_epochs, input_epochs[-1] + step)

    return input_epochs, output_epochs, num_epochs, proj_time


@dataclass(eq=False, kw_only=True)
class NonlinearProjector:
    """
    Object to project a dynamic model forward in time with known inputs.

    This object contstucts a projection object which can simulate a dynamic
    model forward a given amount of time with inputs know a-priori.

    """

    step: float
    """Step size of simulation."""

    proj_time: float
    """How far into the future to project model."""

    num_epochs: int
    """Number of simulation epochs."""

    input_epochs: np.ndarray
    """Epochs at which inputs are applied to model."""

    output_epochs: np.ndarray
    """Epochs at which output results occur."""

    _rep_step: np.ndarray = field(init=False, repr=False)
    _mapped_dyn: ca.Function = field(init=False, repr=False)
    _c_func: ca.Function = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """
        Calculate the simulation epochs from total projection time and sim step.

        Parameters
        ----------
        proj_time: float
            Time to project model forward
        step: float
            Step between simulation epochs
        num_epochs: int
            Number of simulated epochs

        """
        self._rep_step = self.step * np.ones((self.num_epochs,))

    @classmethod
    def create_from_integrator(
        cls: type[T],
        integrator: ca.Function,
        step: float,
        num_epochs: int = None,
        proj_time: float = None,
    ) -> T:
        input_epochs, output_epochs, num_epochs_proc, proj_time_proc = calculate_epochs(
            proj_time=proj_time, num_epochs=num_epochs, step=step
        )

        new_obj = cls(
            step=step,
            num_epochs=num_epochs_proc,
            proj_time=proj_time_proc,
            input_epochs=input_epochs,
            output_epochs=output_epochs,
        )

        new_obj._mapped_dyn = integrator.mapaccum("mapped_int", new_obj.num_epochs)

        return new_obj

    @classmethod
    def load_from_file(cls: T, so_path: Path) -> T:
        pkl_path = so_path.with_suffix(".pkl")
        with open(pkl_path, "rb") as pkl_file:
            pkl_data = pickle.load(pkl_file)

        new_obj = cls(**pkl_data)

        new_obj._mapped_dyn = ca.external("mapped_int", str(so_path))

        return new_obj

    def project_forward_map(
        self,
        *,
        init_t_s: float,
        init_states: na.NamedVector,
        input_array: na.NamedArray,
        input_times: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project dynamic model forward, saving intermediate steps.

        The given input array must be compatible with the model descibed in the
        integrator object used to construct the projector object. The inputs in
        the input_array are interpolated to get the inputs at each simulation
        epoch using the input_times as the x-reference coordinate.

        Parameters
        ----------
        init_t_s: float
            Time corresponding to initial states
        init_states: na.NamedVector
            Initial states to simulation from
        input_array: na.NamedArray
            Array of inputs to apply to model
        input_times: ArrayLike
            Reference times at which given input_array should be applied

        Returns
        -------
        states: np.ndarray
            2-D array of states at each simulation epoch, rows represent states
        epochs: np.ndarray
            1-D array of epochs corresponding to projected states

        """
        inputs = input_array.interp1d(x=self.input_epochs + init_t_s, xp=input_times)

        casadi_result = self._mapped_dyn(
            init_states.to_casadi_array(), inputs, self._rep_step
        )

        input_states_col = np.reshape(init_states.to_numpy_array(), (-1, 1))
        return (
            np.hstack((input_states_col, np.array(casadi_result))),
            self.output_epochs + init_t_s,
        )

    def project_forward(
        self,
        *,
        init_t_s: float,
        init_states: na.NamedVector,
        input_array: na.NamedArray,
        input_times: ArrayLike,
    ) -> tuple[na.NamedVector, float]:
        """
        Project dynamic model forward, returning final result.

        The given input array must be compatible with the model descibed in the
        integrator object used to construct the projector object. The inputs in
        the input_array are interpolated to get the inputs at each simulation
        epoch using the input_times as the x-reference coordinate.

        Parameters
        ----------
        init_t_s: float
            Time corresponding to initial states
        init_states: na.NamedVector
            Initial states to simulation from
        input_array: na.NamedArray
            Array of inputs to apply to model
        input_times: ArrayLike
            Reference times at which given input_array should be applied

        Returns
        -------
        states: na.NamedVector
            NamedVector with same type as init_states with results at end of simulation
        epoch: float
            Epoch corresponding to projected states

        """
        state_type = type(init_states)

        result, out_epochs = self.project_forward_map(
            init_t_s=init_t_s,
            init_states=init_states,
            input_array=input_array,
            input_times=input_times,
        )

        return state_type.from_array(result[:, -1]), out_epochs[-1]

    def pickle(self, pkl_path: Path):
        out_dict = {
            "step": self.step,
            "proj_time": self.proj_time,
            "num_epochs": self.num_epochs,
            "input_epochs": self.input_epochs,
            "output_epochs": self.output_epochs,
        }

        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(obj=out_dict, file=pkl_file)

    def generate_c_code(
        self,
        output_dir: Path,
        output_name: str,
    ) -> None:

        print(f"Generating projector {output_name}...", end="", flush=None)

        pickle_name = Path(output_name).with_suffix(".pkl")
        c_name = Path(output_name).with_suffix(".c")
        so_name = Path(output_name).with_suffix(".so")

        cwd = Path.cwd()
        try:
            os.chdir(output_dir)
            self._mapped_dyn.generate(str(c_name))
            os.system(f"gcc -fPIC -O3 -shared {c_name} -o {so_name}")

            os.remove(c_name)  # Clean up intermediate .c file
            self.pickle(pickle_name)
        finally:
            os.chdir(cwd)

        print("Done!")
