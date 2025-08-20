# src/problems/allen_cahn_equation.py

import numpy as np
import torch
import deepxde as dde
import os
from scipy.io import loadmat
from .base_problem import BaseProblem

class AllenCahnEquationProblem(BaseProblem):
    """
    Defines the 1D Allen-Cahn Equation problem.
    PDE: u_t - d * u_xx - 5 * (u - u^3) = 0
    Domain: x in [-1, 1], t in [0, 1]
    IC: u(x, 0) = x^2 * cos(pi * x)
    BC: u(-1, t) = -1, u(1, t) = -1
    """
    def __init__(self, config):
        problem_cfg = config['problem']
        self.L_min = -1.0
        self.L_max = 1.0
        self.T = 1.0
        self.d = problem_cfg.get('d', 0.001)
        self.ic_bcs_config = problem_cfg.get('ic_bcs', {'ic': True, 'bc': True})
        
        super().__init__(config)
        self.plot_amplitude = 1.2

        # Load discrete ground truth data for final evaluation
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
        u_xx = dde.grad.hessian(u, inputs, i=0, j=0)
        return u_t - self.d * u_xx - 5 * (u - u**3)

    def get_initial_condition_func(self):
        return lambda x: x[:, 0:1]**2 * np.cos(np.pi * x[:, 0:1])
        
    def get_boundary_condition_func(self):
        return lambda x: -1.0

    def get_ics_bcs(self):
        """Returns the list of ICs and BCs for soft-constraint models."""
        ic_bcs_list = []
        if self.ic_bcs_config.get('bc', True):
            # Use a function for the value to handle the -1 BC
            bc = dde.icbc.DirichletBC(
                self.geomtime, 
                self.get_boundary_condition_func(), 
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
        """
        Hard-constraint output transform from the DeepXDE demo.
        Enforces both IC and BC.
        """
        def output_transform(inputs, outputs):
            x, t = inputs[:, 0:1], inputs[:, 1:2]
            # When t=0, this is x^2*cos(pi*x) + 0 = IC
            # When x= +/-1, this is 1*cos(pi) + t*(1-1)*NN = -1 = BC
            return x**2 * torch.cos(np.pi * x) + t * (1 - x**2) * outputs
        return output_transform

    def _load_ground_truth_data(self, path):
        print(f"Loading discrete ground truth data from {path} for final evaluation...")
        data = loadmat(path)
        t, x, u = data["t"], data["x"], data["u"]
        xx, tt = np.meshgrid(x, t)
        self.X_test = np.vstack((np.ravel(xx), np.ravel(tt))).T
        self.y_test = u.flatten()[:, None]
        print(f"Loaded {self.X_test.shape[0]} test points for final evaluation.")

    def get_test_data(self):
        return self.X_test, self.y_test

    def analytical_solution(self, xt):
        """Dummy function for models without hard constraints."""
        return np.zeros((xt.shape[0], 1))
            
    def get_plot_amplitude(self):
        return self.plot_amplitude