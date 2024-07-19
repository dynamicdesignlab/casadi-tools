""" NLP Runner Module

This module implements a runner class for nonlinear programming problems
in casadi

"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, NamedTuple
from typing_extensions import Self

import casadi as ca
import numpy as np

from casadi_tools.nlp_utils import nlp_problem

STATUS_TO_EXIT_FLAG = {
    "Solve_Succeeded": 1,
    "Solved_To_Acceptable_Level": 2,
    "User_Requested_Stop": 3,
    "Feasible_Point_Found": 4,
    "Maximum_Iterations_Exceeded": -1,
    "Restoration_Failed": -2,
    "Error_In_Step_Computation": -3,
    "Maximum_CpuTime_Exceeded": -4,
    "Infeasible_Problem_Detected": -5,
    "Search_Direction_Becomes_Too_Small": -6,
    "Diverging_Iterates": -7,
    "Not_Enough_Degrees_Of_Freedom": -10,
    "Invalid_Problem_Definition": -11,
    "Invalid_Option": -12,
    "Unrecoverable_Exception": -100,
    "NonIpopt_Exception_Thrown": -101,
    "Insufficient_Memory": -102,
    "Internal_Error": -199,
}
EXIT_FLAG_TO_STATUS = {value: key for key, value in STATUS_TO_EXIT_FLAG.items()}


class NLPSolveStats(NamedTuple):
    iterations: int
    exit_flag: int
    solve_time_ms: float
    objective: float


@dataclass
class NLPRunner:
    nlp: nlp_problem.NLPProblem
    solver: ca.Function = field(repr=False)
    dual_var_warmstart: bool  # Note: this is False by default in the load_nlp function

    _lam_x_vector: np.ndarray = field(init=False, repr=False, default=None)
    _lam_g_vector: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.update_init_guess()

    def update_init_guess(self) -> None:
        """Update runner initial guess from NLP problem member instance."""
        self._lam_x_vector = np.zeros(self.nlp.lbx_vector.shape)
        self._lam_g_vector = np.zeros(self.nlp.lbg_vector.shape)

    def run_solver(self) -> tuple[dict[str, np.ndarray], NLPSolveStats]:
        """
        Run NLP solver.

        Run the contained NLP solver and return the results and relevant
        solve stats.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of results
        NLPSolveStats
            NamedTuple of solver stats

        """
        if self.nlp.params.param_left_unset():
            unset_params = self.nlp.params.get_unset_parameters()
            raise ValueError(f"Parameters left unset: {unset_params}")

        solver_args = {
            "x0": self.nlp.init_guess_vector,
            "p": self.nlp.params.value_vector,
            "lbx": self.nlp.lbx_vector,
            "ubx": self.nlp.ubx_vector,
            "lbg": self.nlp.lbg_vector,
            "ubg": self.nlp.ubg_vector,
        }

        if self.dual_var_warmstart:
            solver_args["lam_x0"] = self._lam_x_vector
            solver_args["lam_g0"] = self._lam_g_vector

        start_time = time.perf_counter()
        res = self.solver(**solver_args)
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
        raw_stats = self.solver.stats()

        exit_flag = STATUS_TO_EXIT_FLAG.get(raw_stats["return_status"], -999)
        out_vec = res["x"].full().squeeze()

        if exit_flag == 1:
            self.nlp.update_guess_from_result_vector(out_vec)
            self._lam_x_vector = res["lam_x"].full().squeeze()
            self._lam_g_vector = res["lam_g"].full().squeeze()

        stats = NLPSolveStats(
            iterations=raw_stats["iter_count"],
            objective=float(res["f"]),
            exit_flag=exit_flag,
            solve_time_ms=elapsed_time_ms,
        )

        return self.nlp.unpack_results(out_vec), stats
    
    def update_solver(self, so_file, solver_opts={}, solver_name='ipopt'):
        """Update solver for current NLPRunner instance."""
        
        self.solver = ca.nlpsol("nlpsol", solver_name, str(so_file), solver_opts)
        
        return  

    @classmethod
    def load_nlp(
        cls,
        so_file: Path,
        *,
        solver_name: str = "ipopt",
        solver_opts: Dict[str, Any] = None,
        dual_var_warmstart: bool = False,
    ) -> Self:
        """
        Create NLPRunner from saved solver assets.

        Load a solver shared object and it's associated pickled
        NLPProblem and from them create a NLPRunner instance.

        Parameters
        ----------
        so_file: Path
            Path to solver shared object
        solver_name: str = "ipopt"
            Name of casadi solver to use
        solver_opts: Dict[str, Any], optional
            Dictionary of solver options

        """
        if solver_opts is None:
            solver_opts = {}

        solver = ca.nlpsol("nlpsol", solver_name, str(so_file), solver_opts)

        nlp_pickle = so_file.with_suffix(".pkl")
        with open(nlp_pickle, "rb") as pkl_file:
            nlp = pickle.load(pkl_file)

        return cls(nlp=nlp, solver=solver, dual_var_warmstart=dual_var_warmstart)
