"""Implements a runner class for nlp's in casadi."""

import enum
import os
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List

import casadi as ca
import numpy as np

from casadi_tools import types
from casadi_tools.nlp_utils.nlp_elems import ConstraintVariable as CV
from casadi_tools.nlp_utils.nlp_elems import DecisionVariable as DV
from casadi_tools.nlp_utils.nlp_elems import ParameterVariable as PV


class NLPHorizonError(Exception):
    pass


class OptType(enum.Enum):
    NLP = enum.auto
    QP = enum.auto


@dataclass
class NLPProblem:
    horizons: tuple[np.ndarray]
    """ Prediction horizons for each multistage """

    states: DV = field(init=False, repr=False, default_factory=DV.factory)
    """ State decision variable object """
    inputs: DV = field(init=False, repr=False, default_factory=DV.factory)
    """ Input decision variable object """
    other: DV = field(init=False, repr=False, default_factory=DV.factory)
    """ Slack decision variable object """

    cstrs: CV = field(init=False, repr=False, default_factory=CV.factory)
    """ Inequality constraints variable object """

    params: PV = field(init=False, repr=False, default_factory=PV.factory)
    """ Parameter variable object """

    _objective: types.CASADI_TYPE = field(init=False, repr=False, default=None)

    @property
    def num_horizons(self) -> int:
        """Number of unique horizon problems NLPProblem contains."""
        return len(self.horizons)

    def num_stages(self, horizon_idx: int = None) -> int:
        """
        Number of stages in prediction horizon.

        If no horizon index is provided, all horizon stages will be returned as
        a tuple.

        Parameters
        ----------
        horizon_idx: int, optional
            Index of horizon to query

        Returns
        -------
        int
            Number of stages in queried horizon

        Raises
        ------
        NLPHorizonError
            If the horizon_idx is below zero or greater than the largest index

        """
        if horizon_idx is None:
            return tuple([x.size for x in self.horizons])

        if horizon_idx < 0 or horizon_idx > self.num_horizons - 1:
            raise NLPHorizonError("Invalid horizon index.")

        return self.horizons[horizon_idx].size

    def steps(self, horizon_idx: int = None) -> np.ndarray:
        """
        Steps in prediction horizon (difference between stages).

        Returns an array of size N-1 if the horizon is length N.

        If no horizon index is provided, all horizon steps will be returned as
        a tuple.

        Parameters
        ----------
        horizon_idx: int, optional
            Index of horizon to query

        Returns
        -------
        np.ndarray
            Step between each horizon stage

        Raises
        ------
        NLPHorizonError
            If the horizon_idx is below zero or greater than the largest index

        """
        if horizon_idx is None:
            return tuple([np.diff(x) for x in self.horizons])

        if horizon_idx < 0 or horizon_idx > self.num_horizons - 1:
            raise NLPHorizonError("Invalid horizon index.")

        return np.diff(self.horizons[horizon_idx])

    def horizon_dict(
        self, *, init_time: float = 0.0, scale: tuple[float] = None
    ) -> dict[str, np.ndarray]:
        """Dict containing each prediciton horizon of the multistage problem."""
        horizons = {}

        if scale is None:
            scale = [1.0 for _ in range(self.num_horizons)]
        elif not len(scale) == self.num_horizons:
            raise ValueError("Number of scale factors does not match number of stages.")

        for idx, val in enumerate(scale):
            new_horz = self.single_horizon(idx, init_time=init_time, scale=val)
            horizons[f"time__{idx}"] = new_horz
            init_time = new_horz[..., -1]

        return horizons

    def full_horizon(
        self, *, init_time: float = 0.0, scale: tuple[float] = None
    ) -> np.ndarray:
        """Array containing each stage of the prediciton horizon."""
        horizons = []

        if scale is None:
            scale = [1.0 for _ in range(self.num_horizons)]
        elif not len(scale) == self.num_horizons:
            raise ValueError("Number of scale factors does not match number of stages.")

        for idx, val in enumerate(scale):
            new_horz = self.single_horizon(idx, init_time=init_time, scale=val)
            horizons.append(new_horz)
            init_time = new_horz[..., -1]

        return np.concatenate(horizons, axis=1)

    def single_horizon(
        self, stage_num: int, *, init_time: float = 0.0, scale: float = 1.0
    ) -> np.ndarray:
        """Return array of horizon steps for one of the multistage problems."""
        return init_time + scale * self.horizons[stage_num].reshape((1, -1))

    @property
    def num_decision_var(self) -> int:
        return self.states.num_fields + self.inputs.num_fields + self.other.num_fields

    @property
    def num_decision_elem(self) -> int:
        return self.states.num_elem + self.inputs.num_elem + self.other.num_elem

    @property
    def decision_names(self) -> List[str]:
        return (
            self.states.field_names + self.inputs.field_names + self.other.field_names
        )

    @property
    def decision_vector(self) -> types.CASADI_TYPE:
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
        return ca.horzcat(
            self.states.to_vector(), self.inputs.to_vector(), self.other.to_vector()
        )

    @property
    def lbx_vector(self) -> np.ndarray:
        return np.concatenate(
            (self.states.lb_vector, self.inputs.lb_vector, self.other.lb_vector)
        )

    @property
    def ubx_vector(self) -> np.ndarray:
        return np.concatenate(
            (self.states.ub_vector, self.inputs.ub_vector, self.other.ub_vector)
        )

    @property
    def init_guess_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.states.init_guess_vector,
                self.inputs.init_guess_vector,
                self.other.init_guess_vector,
            )
        )

    @property
    def init_lam_g_vector(self) -> np.ndarray:
        return np.zeros(self.states.lb_vector.shape)

    @property
    def lbg_vector(self) -> np.ndarray:
        return self.cstrs.lb_vector

    @property
    def ubg_vector(self) -> np.ndarray:
        return self.cstrs.ub_vector

    @property
    def objective(self) -> types.CASADI_TYPE:
        return self._objective

    def unpack_results(self, results_vector: np.ndarray) -> dict[str, np.ndarray]:
        """
        Unpack 1-D result vector from solver into dictionary.

        Parameters
        ----------
        results_vector: types.CASADI_TYPE
            Vector of results obtained from nlp solver

        Returns
        -------
        results: dict[str, np.ndarray]:
            Dictionary of decision variable names to result arrays

        """
        results = {}
        lo_idx = 0
        for elem in (self.states, self.inputs, self.other):
            for key in elem.field_names:
                hi_idx = lo_idx + elem.get_size(key)
                results[key] = results_vector[lo_idx:hi_idx].reshape((1, -1))
                lo_idx = hi_idx

        return results

    def update_guess_from_result_vector(
        self, results_vector: types.CASADI_TYPE
    ) -> None:
        """
        Update guesses of all nlp elements based on results of last solve.

        Parameters
        ----------
        results_vector: types.CASADI_TYPE
            Vector of results obtained from nlp solver
        """
        result_dict = self.unpack_results(results_vector=results_vector)

        for elem in (self.states, self.inputs, self.other):
            for key in elem.field_names:
                elem.set_init_guess(key, result_dict[key])

    def add_objective(self, expr: types.CASADI_TYPE) -> None:
        """
        Add objective to nlp problem.

        Add an objective to the problem formulation. The output must be a
        scalar casadi symbolic. This will additively combine with any previously
        added objectives. If this is called before any other objective is set,
        it will set the objective to the expr parameter.


        Parameters
        ----------
        expr : MX or SX or DM
            Casadi symbolic expression representing the objective function

        """
        if self._objective is None:
            self.set_objective(expr)
        else:
            self._objective = self._objective + expr

    def set_objective(self, expr: types.CASADI_TYPE) -> None:
        """
        Set objective for nlp problem.

        Overwrites current objective with expression

        Parameters
        ----------
        expr : MX or SX or DM
            Casadi symbolic expression representing the objective function

        """
        self._objective = expr

    def pickle(self, pickle_file: Path) -> None:
        new_self = deepcopy(self)

        new_self.states.clear_expr()
        new_self.inputs.clear_expr()
        new_self.other.clear_expr()
        new_self.params.clear_expr()
        new_self.cstrs.clear_expr()
        new_self._objective = None

        with open(pickle_file, "wb") as pkl:
            pickle.dump(new_self, pkl)


def generate_solver(
    name: str,
    *,
    nlp: NLPProblem,
    opt_type: OptType,
    solver: str,
    opts: dict = None,
) -> ca.Function:
    """
    Generate solver object.

    Generate a Casadi nlpsol function based on the objects contained within
    the NLPBuilder instance. The interface is the same as the nlpsol object
    within casadi. Refer to the casadi docs on what options are available
    for the solver and options fields, and how to call the solver object
    after it is generated

    Parameters
    ----------
    name : str
        Name of the solver object
    nlp: NLPProblem
        NLPProblem object to generate solver from
    opt_type: OptType
        Type of optimization problem (NLP or QP)
    solver : str
        Name of the nlp solver to use, e.g. 'ipopt'
    opts: dict, optional
        Dictionary of options for casadi and the underlying solver

    """
    if opts is None:
        opts = {}

    nlp_dict = {
        "x": nlp.decision_vector,
        "f": nlp.objective,
        "g": nlp.cstrs.to_vector(),
        "p": nlp.params.to_vector(),
    }

    if opt_type == OptType.NLP:
        return ca.nlpsol(name, solver, nlp_dict, opts)
    elif opt_type == OptType.QP:
        return ca.qpsol(name, solver, nlp_dict, opts)
    else:
        raise ValueError("Invalid optimization type.")


def generate_shared_object(
    *,
    solver: ca.Function,
    nlp: NLPProblem,
    output_dir: Path,
    output_name: str,
    opt_level: int = 0,
) -> None:
    """
    Generate shared object from solver.

    Convenience function to use the casadi/nlpsol framework to generate .c
    code and then gcc to generate a shared object library containing your
    solver.

    This function also saves a pickled nlp descriptor dictionary
    with the same name as the shared object.

    Parameters
    ----------
    solver : casadi.Function
        Casadi NLPSOL object
    nlp: NLPProblem
        NLPProblem instance to generate shared object from
    output_dir: Path
        Directory to save generated shared object in
    output_name : str
        Name of the compiled shared object
    opt_level: int {0, 1, 2, 3}
        Level of gcc optimization for to compile the object with

    See Also
    --------
    pack_nlp_as_dict : Function to generate nlp descriptor dictionary

    """
    if (opt_level > 3) or (opt_level < 0):
        raise (
            ValueError,
            "Optimization level must be between 0 and 3," "inclusive",
        )

    print(
        f'Generating library "{output_name}.so",',
        f"with optimization level: -O{opt_level}\n",
    )
    print(f"Began compiling at: {time.ctime()}... ", end="", flush=True)

    pickle_name = Path(output_name).with_suffix(".pkl")
    c_name = Path(output_name).with_suffix(".c")
    so_name = Path(output_name).with_suffix(".so")

    cwd = Path.cwd()
    start_time = time.time()
    try:
        os.chdir(output_dir)
        solver.generate_dependencies(str(c_name))
        os.system(f"gcc -fPIC -O{opt_level} -shared {c_name} -o {so_name}")

        os.remove(c_name)  # Clean up intermediate .c file
        nlp.pickle(pickle_name)
    finally:
        os.chdir(cwd)
    end_time = time.time()

    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60.0

    print("Done!")
    print(f"Elapsed time: {elapsed_min:.1f} min ({elapsed_sec:.1f} sec)\n\n")


def create_nlp(
    state_names: Iterable[str],
    input_names: Iterable[str],
    horizon: np.ndarray,
) -> NLPProblem:
    nlp = NLPProblem(
        horizons=(horizon,),
    )

    for state_name in state_names:
        nlp.states.add_field(state_name, nlp.num_stages(0))

    for input_name in input_names:
        nlp.inputs.add_field(input_name, nlp.num_stages(0))

    return nlp


def create_nlp_with_dynamics(
    state_names: Iterable[str],
    input_names: Iterable[str],
    dynamics: ca.Function,
    horizon: np.ndarray,
    create_init_cstr: bool = True,
) -> NLPProblem:
    """
    Create NLP with dynamics.

    Create an NLPBuilder object from a dynamic model. This function will
    automatically generate the states and inputs needed.

    Additionally, this function creates appropriate initial state parameters
    and then generates dynamic constraints over the nlp horizon.

    The NLPBuilder object is returned so the user can then specify the
    objective and create other attributes if desired.

    Parameters
    ----------
    state_names: Iterable[str]
        Names of state variables
    input_names: Iterable[str]
        Names of input variables
    dynamics: ca.Function
        Method or function returning next set of states given currents states and inputs
    horizon : np.ndarray
        Stage array for NLP problem


    Returns
    -------
    NLPBuilder
        NLPBuilder object pre-populated with relevant attributes

    """
    nlp = create_nlp(
        state_names=state_names,
        input_names=input_names,
        horizon=horizon,
    )

    if create_init_cstr:
        generate_init_state_constr(nlp)

    generate_dyn_constr(nlp, dynamics, nlp.states.to_array(), nlp.inputs.to_array())

    return nlp


def generate_dyn_constr_params(
    nlp: NLPProblem,
    dynamics: Callable,
    states: types.CASADI_TYPE,
    inputs: types.CASADI_TYPE,
    other: types.CASADI_TYPE,
) -> None:
    
    old_states = states[:, :-1]
    old_inputs = inputs[:, :-1]

    new_states = states[:, 1:]
    new_inputs = inputs[:, 1:]
    
    if (other.shape[1] == states.shape[1]):
        old_other = other[:, :-1]
        new_other = other[:, 1:]
    elif (other.shape[1] == 1):
        old_other = other
        new_other = other
    else:
        raise ValueError("Parameters to dynamics is an incompatible size.")

    dt_N = nlp.steps(0)

    map_integ = dynamics.map(nlp.num_stages(0) - 1)
    if dynamics.name() == "trapz":
        calc_states = map_integ(
            old_other, new_other, old_states, new_states, old_inputs, new_inputs, dt_N
        )
    else:
        calc_states = map_integ(old_other, old_states, old_inputs, dt_N)

    dyn_g_vec = new_states - calc_states
    for name, g in zip(nlp.states.field_names, ca.vertsplit(dyn_g_vec)):
        nlp.cstrs.add_field(f"dyn_{name}", g, lower=0.0, upper=0.0)


def generate_dyn_constr(
    nlp: NLPProblem,
    dynamics: Callable,
    states: types.CASADI_TYPE,
    inputs: types.CASADI_TYPE,
) -> None:
    old_states = states[:, :-1]
    old_inputs = inputs[:, :-1]

    new_states = states[:, 1:]
    new_inputs = inputs[:, 1:]

    dt_N = nlp.steps(0)

    map_integ = dynamics.map(nlp.num_stages(0) - 1)
    if dynamics.name() == "trapz":
        calc_states = map_integ(old_states, new_states, old_inputs, new_inputs, dt_N)
    else:
        calc_states = map_integ(old_states, old_inputs, dt_N)

    dyn_g_vec = new_states - calc_states
    for name, g in zip(nlp.states.field_names, ca.vertsplit(dyn_g_vec)):
        nlp.cstrs.add_field(f"dyn_{name}", g, lower=0.0, upper=0.0)


def generate_init_state_constr(nlp: NLPProblem) -> None:
    nlp.params.add_field("init_states", nlp.states.num_fields)
    states = nlp.states.to_array()
    init_g_vec = states[:, 0] - nlp.params.get_expr("init_states")
    nlp.cstrs.add_field("init_states", init_g_vec, lower=0.0, upper=0.0)


def merge_nlps(nlps: Iterable[NLPProblem], *, link_stages=False) -> NLPProblem:
    horizons = []
    for prob in nlps:
        horizons += [x for x in prob.horizons]

    new_prob = NLPProblem(horizons=tuple(horizons))

    state_list = [prob.states for prob in nlps]
    input_list = [prob.inputs for prob in nlps]
    new_prob.states = DV.merge(state_list)
    new_prob.inputs = DV.merge(input_list)
    new_prob.other = DV.merge([prob.other for prob in nlps])

    new_prob.cstrs = CV.merge([prob.cstrs for prob in nlps])

    new_prob.params = PV.merge([prob.params for prob in nlps])

    for prob in nlps:
        new_prob.add_objective(prob._objective)

    if link_stages:
        _link_stages(new_nlp=new_prob, nlp_states=state_list)

    return new_prob


def _find_num_multistage(var_names: Iterable[str]) -> int:
    digit_multistage = [name.rsplit(sep="__", maxsplit=1) for name in var_names]
    int_multistage = []
    for val in digit_multistage:
        try:
            _, value = val
        except ValueError:
            continue
        int_multistage.append(int(value))
    return max(int_multistage) + 1


def _check_vars_compatible(var_names: Iterable[str]) -> bool:
    names = [set() for _ in range(_find_num_multistage(var_names))]
    for name in var_names:
        try:
            base_name, idx = name.rsplit(sep="__", maxsplit=1)
        except ValueError:
            return False
        names[int(idx)].add(base_name)

    for name_set in names:
        if not name_set == names[0]:
            return False

    return True


def _link_stages(new_nlp: NLPProblem, nlp_states: Iterable[DV]) -> None:
    states_good = _check_vars_compatible(new_nlp.states.field_names)
    inputs_good = _check_vars_compatible(new_nlp.inputs.field_names)
    if not states_good or not inputs_good:
        raise RuntimeError("NLP variables are not compatible.")

    num_links = len(nlp_states) - 1
    for idx in range(num_links):
        last_state = nlp_states[idx].to_array()[:, -1]
        first_state = nlp_states[idx + 1].to_array()[:, 0]
        new_nlp.cstrs.add_field(
            name=f"link_{idx}", expr=first_state - last_state, lower=0.0, upper=0.0
        )


def constant_horizon(*, num_stages: int, step_size: float) -> np.ndarray:
    """
    Generate an array of prediction horizon stages using a constant step.

    The prediction horizon begins at stage 0.0, and goes up until (but
    not including) num_stages*step_size.

    Parameters
    ----------
    num_stages: int
        Number of horizon stages
    step_size: float
        Step of size between stages

    Returns
    -------
    np.ndarray
        Array representing each stage of the prediction horizon

    Examples
    --------
    >>> constant_horizon(num_stages=3, step_size=0.1)
    array([0. , 0.1, 0.2])

    """
    return np.linspace(0.0, num_stages * step_size, num_stages, endpoint=False)
