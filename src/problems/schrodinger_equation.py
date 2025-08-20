# src/problems/schrodinger_equation.py

import numpy as np
import torch
import deepxde as dde
import os
from scipy.io import loadmat
from .base_problem import BaseProblem

class SchrodingerEquationProblem(BaseProblem):
    """
    Defines the 1D non-linear Schrödinger (NLS) equation.
    PDE: i*h_t + 0.5*h_xx + |h|^2*h = 0, where h = u + iv
    This is split into two real-valued PDEs for u and v.
    - Real part:  u_t + 0.5*v_xx + (u^2 + v^2)*v = 0
    - Imag part:  v_t - 0.5*u_xx - (u^2 + v^2)*u = 0
    
    Domain: x in [-5, 5], t in [0, pi/2]
    IC: h(0, x) = 2*sech(x)  => u(0,x)=2*sech(x), v(0,x)=0
    BC: Periodic
    """
    def __init__(self, config):
        problem_cfg = config['problem']
        self.L_min = -5.0
        self.L_max = 5.0
        self.T = np.pi / 2
        
        super().__init__(config)
        self.plot_amplitude = 2.2

        self.X_test = None
        self.y_test = None
        ground_truth_path = problem_cfg.get('ground_truth_path')
        if ground_truth_path and os.path.exists(ground_truth_path):
            self._load_ground_truth_data(ground_truth_path)
        else:
            print(f"Warning: Ground truth file not found at '{ground_truth_path}'.")

    @property
    def x_min(self): return self.L_min
    @property
    def x_max(self): return self.L_max

    def setup_domain(self):
        geom = dde.geometry.Interval(self.L_min, self.L_max)
        timedomain = dde.geometry.TimeDomain(0, self.T)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def pde(self, inputs, outputs):
        """
        inputs: x[:, 0] is x-coordinate, x[:, 1] is t-coordinate
        outputs: y[:, 0] is u(x,t), y[:, 1] is v(x,t)
        """
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]

        u_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        v_t = dde.grad.jacobian(outputs, inputs, i=1, j=1)
        
        u_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
        v_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)

        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
        
        return [f_u, f_v]

    def get_ics_bcs(self):
        # Initial Conditions
        ic_u = dde.icbc.IC(
            self.geomtime, 
            lambda x: 2 / np.cosh(x[:, 0:1]), 
            lambda _, on_initial: on_initial, 
            component=0
        )
        ic_v = dde.icbc.IC(
            self.geomtime, 
            lambda x: 0.0, 
            lambda _, on_initial: on_initial, 
            component=1
        )

        # Periodic Boundary Conditions for u, u_x, v, v_x
        bc_u_0 = dde.icbc.PeriodicBC(self.geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
        bc_u_1 = dde.icbc.PeriodicBC(self.geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0)
        bc_v_0 = dde.icbc.PeriodicBC(self.geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
        bc_v_1 = dde.icbc.PeriodicBC(self.geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1)
        
        return [ic_u, ic_v, bc_u_0, bc_u_1, bc_v_0, bc_v_1]

    def _load_ground_truth_data(self, path):
        print(f"Loading ground truth data from {path}...")
        data = loadmat(path)
        t = data['tt'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        Exact_h = data['uu'] # This is complex-valued
        
        # Create meshgrid
        X, T = np.meshgrid(x, t)
        
        # Reshape for testing
        self.X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        
        # Split complex solution into real and imaginary parts
        u_true = np.real(Exact_h).T.flatten()[:,None]
        v_true = np.imag(Exact_h).T.flatten()[:,None]
        self.y_test = np.hstack((u_true, v_true))
        print(f"Loaded {self.X_test.shape[0]} test points.")

    def get_initial_condition_func(self):
        return None
    def get_test_data(self): return self.X_test, self.y_test
    def analytical_solution(self, xt): return np.zeros((xt.shape[0], 2))
    def get_plot_amplitude(self): return self.plot_amplitude
