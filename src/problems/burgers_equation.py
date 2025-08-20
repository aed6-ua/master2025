# src/problems/burgers_equation.py (Corrected and Simplified)

import numpy as np
import torch
import deepxde as dde
import os
from .base_problem import BaseProblem

class BurgersEquationProblem(BaseProblem):
    """
    Defines the 1D Viscous Burgers' Equation problem.
    Loads discrete ground truth data for final evaluation only.
    """
    def __init__(self, config):
        problem_cfg = config['problem']
        self.L_min = problem_cfg.get('L_min', -1.0)
        self.L_max = problem_cfg.get('L_max', 1.0)
        self.T = problem_cfg.get('T', 1.0)
        self.nu = problem_cfg.get('nu')
        self.ic_bcs_config = problem_cfg.get('ic_bcs', {'ic': True, 'bc': True})

        super().__init__(config)
        self.plot_amplitude = 1.1

        # Load and store the discrete ground truth data for final evaluation
        self.X_test = None
        self.y_test = None
        ground_truth_path = problem_cfg.get('ground_truth_path')
        if ground_truth_path and os.path.exists(ground_truth_path):
            self._load_ground_truth_data(ground_truth_path)
        else:
            print(f"Warning: Ground truth file not found at '{ground_truth_path}'. Final evaluation will be skipped.")

    @property
    def x_min(self):
        return self.L_min

    @property
    def x_max(self):
        return self.L_max

    def setup_domain(self):
        geom = dde.geometry.Interval(self.L_min, self.L_max)
        timedomain = dde.geometry.TimeDomain(0, self.T)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def pde(self, inputs, u):
        u_t = dde.grad.jacobian(u, inputs, i=0, j=1)
        u_x = dde.grad.jacobian(u, inputs, i=0, j=0)
        u_xx = dde.grad.hessian(u, inputs, i=0, j=0)
        return u_t + u * u_x - self.nu * u_xx

    def get_initial_condition_func(self):
        return lambda x: -np.sin(np.pi * x[:, 0:1])

    def get_ics_bcs(self):
        ic_bcs_list = []
        if self.ic_bcs_config.get('bc', True):
            bc = dde.icbc.DirichletBC(
                self.geomtime, 
                lambda x: 0.0, 
                lambda x, on_boundary: on_boundary and not dde.utils.isclose(x[1], 0.0)
            )
            ic_bcs_list.append(bc)
        if self.ic_bcs_config.get('ic', True):
            ic = dde.icbc.IC(
                self.geomtime,
                self.get_initial_condition_func(),
                lambda _, on_initial: on_initial
            )
            ic_bcs_list.append(ic)
        return ic_bcs_list
        
    def get_output_transform(self):
        def output_transform(inputs, outputs):
            x = inputs[:, 0:1]
            boundary_factor = (x - self.L_min) * (x - self.L_max)
            return boundary_factor * outputs
        return output_transform

    def _load_ground_truth_data(self, path):
        """Loads ground truth data from the .npz file."""
        print(f"Loading discrete ground truth data from {path} for final evaluation...")
        data = np.load(path)
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        self.X_test = np.vstack((np.ravel(xx), np.ravel(tt))).T
        self.y_test = exact.flatten()[:, None]
        print(f"Loaded {self.X_test.shape[0]} test points for final evaluation.")

    def get_test_data(self):
        """Returns the loaded test data for final evaluation."""
        return self.X_test, self.y_test

    def analytical_solution(self, xt):
        """
        Dummy function. The ground truth is provided as discrete test data
        and should not be evaluated with this function. Returns zeros to avoid errors.
        For problems with ground truth files, DeepXDE's L2 metric will be inaccurate during training,
        but the final evaluation in run_experiment.py will be correct.
        """
        return np.zeros((xt.shape[0], 1))
            
    def get_plot_amplitude(self):
        return self.plot_amplitude