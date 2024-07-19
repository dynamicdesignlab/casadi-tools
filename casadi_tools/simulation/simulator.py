""" Simulation Module

This module implements functions and classes useful for dynamic simulations.

The factory function :func:`simulation.create_sim` creates a SimRunner object
which handles all of the simulation functionality. The simulation runner is
based on a coroutine. Here is a simple, complete example of a simulation
proportional controller with a mass-spring-damper

.. code-block:: python

    from dynamics.mass_spring_damper_force import Model, States, Inputs
    from dynamics.integrators import create_integrator, euler
    from runners.simulation import create_sim, SimLogger

    END_TIME = 0.5
    SIM_STEP = 0.1
    K = 0.5

    INIT_STATES = States(0.0, 0.0)

    model = Model(m=1.0, k=1.0, b=1.0)
    integ = create_integrator(euler, model, States, Inputs)

    log = SimLogger()
    sim = create_sim(integ, END_TIME, STEP, INIT_STATES, log)

    states = sim.begin_sim() # Begin and prime sim coroutine and get
    initial states

    for event in sim:
        force = -K*states.pos
        inputs = Inputs(force)
        states = sim.take_step(inputs) # Take simulation step

    sim.end_sim() # Close sim coroutine

"""

from dataclasses import dataclass, field
from typing import Callable, Generator, Type
from typing_extensions import Self

import numpy as np
from casadi_tools.dynamics import named_arrays as na


@dataclass
class SimRunner:
    """
    Simulation runner class.

    This class implements functionality needed to run a dynamic simulation
    based on python couroutines. Use the builder function
    :func:`~simulation.create_sim` to create an instance of this class rather
    than creating it manually.


    After creation prime the simulation using the
    :meth:`~simulation.SimRunner.begin_sim` method. See detailed documentation
    there.

    Loop through the classes built in iterator while calling the
    :meth:`~simulation.SimRunner.step` method with the desired system inputs to
    step through the simulation.

    Finally end the simulation with the :meth:`~simulation.SimRunner.end_sim`
    method to clean up and complete the simulation

    """

    end_time: float
    """ End time of simulation """

    time_step: float
    """ Discretization step """

    num_events: int
    """ Number of events in simulation """

    init_states: na.NamedVector
    """ Initial state of system to begin simulation"""

    _integ: Callable
    _coro: Generator = None
    _current_event: int = field(init=False, repr=False)
    _timeline: np.ndarray = field(init=False)
    _state_type: Type[na.NamedVector] = field(init=False)

    def __post_init__(self):
        self._timeline = np.linspace(
            0.0, self.end_time, self.num_events + 1, endpoint=True
        )
        self._state_type = type(self.init_states)

    def __iter__(self):
        return (event for event in range(self.num_events))

    def __len__(self):
        return self.num_events

    @property
    def timeline(self) -> tuple[float]:
        """
        List of event timestamps for simulation.

        Returns
        -------
        tuple[float]
            Tuple of event timestamps

        """
        return self._timeline

    @property
    def current_event(self) -> int:
        return self._current_event

    @property
    def current_time(self) -> float:
        """
        Get time of event.

        Parameters
        ----------
        event: int
            Index of event

        Returns
        -------
        float
            Time at event

        """
        return self.timeline[self.current_event]

    def get_time(self, event: int) -> float:
        """
        Get time of event.

        Parameters
        ----------
        event: int
            Index of event

        Returns
        -------
        float
            Time at event

        """
        return self.timeline[event]

    def begin(self):
        """
        Begin simulation coroutine.

        This method sets up and then primes the simulation coroutine. This
        method must be called before looping through simulation to get
        coroutine handle.

        """
        self._current_event = 0
        self._coro = self._sim_coro()
        return next(self._coro)

    def end(self) -> None:
        """
        End simulation coroutine.

        This method cleans up the simulation coroutine and the pads the input
        log to make it's length match the state log

        """
        self._coro.close()

    def take_step(
        self, inputs: na.NamedVector, params: na.NamedVector = None
    ) -> na.NamedVector:
        """
        Take step in dynamic simulation.

        This method takes a single simulation step and returns the new system
        states. If a logger was included during class instantiation, the states
        and inputs are logged automatically.

        Parameters
        ----------
        inputs: NamedVector
            NamedVector of current inputs


        Returns
        -------
        states: NamedVector
            NamedVector of current dynamic states
        params: NamedVector
            NamedVector of parameters at current step

        """
        return self._coro.send((inputs, params))

    def _sim_coro(
        self,
    ) -> Generator[na.NamedVector, tuple[na.NamedVector, na.NamedVector], None]:
        """
        Step through dynamic simulation.

        This method implements a coroutine to step through a dynamic simulation
        given a dynamic model and numeric integration scheme. The simulation
        will run until the client calls the "close" method on the simulaiton
        coroutine object.


        Receives
        --------
        inputs: NamedVector
            NamedVector of current inputs


        Yields
        ------
        states:
            NamedVector of current states

        """
        states = self.init_states.to_casadi_array()
        while self._current_event <= self.num_events:
            inputs, params = yield self._state_type.from_array(states.full().squeeze())

            try:
                states = self._integ(
                    params, states, inputs.to_casadi_array(), self.time_step
                )
            except RuntimeError:
                states = self._integ(states, inputs.to_casadi_array(), self.time_step)

            self._current_event += 1

    @classmethod
    def create_sim(
        cls,
        integrator: Callable,
        end_time: float,
        step: float,
        init_states: na.NamedVector,
    ) -> Self:
        """
        Build simulation instance.

        This function creates a SimRunner instance given the stated parameters for
        a dynamic simulation

        Parameters
        ----------
        integrator: Callable
            Integrator created from integrators module using
            :func:`dynamics.integrator.create_integrator`
        end_time: float
            Desired end of simulation
        step: float
            Step size of integration between simulation events
        init_states: NamedVector
            Initial states of system
        logger: Logger, default=None
            Logger object

        Returns
        -------
        runner: SimRunner
            Instance of SimRunner

        """
        num_events = int(end_time / step)
        round_end_time = step * num_events

        return cls(
            end_time=round_end_time,
            time_step=step,
            num_events=num_events,
            init_states=init_states,
            _integ=integrator,
        )
