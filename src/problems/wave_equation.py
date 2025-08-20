import numpy as np
import torch
import deepxde as dde
from .base_problem import BaseProblem

class WaveEquationProblem(BaseProblem):
    def __init__(self, config):
        # Extract parameters from the config dictionary
        problem_cfg = config['problem']
        self.L = problem_cfg['L']
        self.T = problem_cfg['T']
        self.c_squared = problem_cfg['c_squared'] # For wave eq: u_tt = c^2 * u_xx.
        
        # Initial condition parameters
        ic_params = problem_cfg.get('ic_params', {})
        self.mode = ic_params.get('mode', 4)
        self.delta = ic_params.get('delta', 10.0)
        self.multi = ic_params.get('multi', False)  # Whether to use multi-scale initial conditions
        self.ic_bcs = problem_cfg.get('ic_bcs', {})

        super().__init__(config)
        self.plot_amplitude = self.get_plot_amplitude()
        print(f"Initialized WaveEquationProblem with c^2 = {self.c_squared}")

    @property
    def x_min(self):
        return 0.0

    @property
    def x_max(self):
        return self.L

    def setup_domain(self):
        geom = dde.geometry.Interval(0, self.L)
        timedomain = dde.geometry.TimeDomain(0, self.T)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def pde(self, inputs, u):
        """Defines the residual of the wave equation: u_tt - c^2 * u_xx = 0"""
        u_tt = dde.grad.hessian(u, inputs, i=1, j=1)
        u_xx = dde.grad.hessian(u, inputs, i=0, j=0)
        return u_tt - self.c_squared * u_xx

    def get_initial_condition_func(self):
        """Returns the function for the initial displacement u(x, 0)."""
        def ic_func_u(x_input_np):
            if isinstance(x_input_np, torch.Tensor):
                if self.multi:
                    return self.delta * torch.sin(self.mode * np.pi * x_input_np[:, 0:1]) + torch.sin(np.pi * x_input_np[:, 0:1])
                return self.delta * torch.sin(self.mode * np.pi / self.L * x_input_np[:, 0:1])
            if self.multi:
                return self.delta * np.sin(self.mode * np.pi * x_input_np[:, 0:1]) + np.sin(np.pi * x_input_np[:, 0:1])
            return self.delta * np.sin(self.mode * np.pi / self.L * x_input_np[:, 0:1])
        return ic_func_u

    def get_ics_bcs(self):
        """Returns the list of ICs and BCs for the wave equation problem."""
        ic_bcs = []
        # 1. Boundary Conditions: u(0, t) = 0 and u(L, t) = 0
        if self.ic_bcs.get('bc', True):
            # Dirichlet BCs at boundaries
            bc = dde.icbc.DirichletBC(
                self.geomtime, 
                lambda x: 0.0, 
                lambda x, on_boundary: on_boundary
            )
            ic_bcs.append(bc)

        # 2. Initial Condition for displacement: u(x, 0) = f(x)
        if self.ic_bcs.get('ic', True):
            ic_u = dde.icbc.IC(
                self.geomtime,
                self.get_initial_condition_func(),
                lambda _, on_initial: on_initial
            )
            ic_bcs.append(ic_u)

        # 3. Initial Condition for velocity: u_t(x, 0) = g(x) (here, g(x)=0)
        # We define this using OperatorBC, which applies a differential operator.
        if self.ic_bcs.get('ic_ut', True):
            ic_ut = dde.icbc.OperatorBC(
                self.geomtime,
                lambda inputs, u, _: dde.grad.jacobian(u, inputs, i=0, j=1), # This is du/dt
                lambda inputs, _: dde.utils.isclose(inputs[1], 0)
            )
            ic_bcs.append(ic_ut)
        
        return ic_bcs

    def analytical_solution(self, xt):
        """Analytical solution for the standing wave initial condition."""
        # Check if input is a torch tensor or numpy array
        is_torch = isinstance(xt, torch.Tensor)
        
        if is_torch:
            x, t = xt[:, 0:1], xt[:, 1:2]
            c = torch.tensor(np.sqrt(self.c_squared), device=xt.device, dtype=xt.dtype)
            _L = torch.tensor(self.L, device=xt.device, dtype=xt.dtype)
            _delta = torch.tensor(self.delta, device=xt.device, dtype=xt.dtype)
            _mode = torch.tensor(self.mode, device=xt.device, dtype=xt.dtype)
            
            # Using torch functions for GPU compatibility
            if self.multi:
                return _delta * torch.sin(_mode * torch.pi * x) * torch.cos(_mode * torch.pi * c * t) + torch.sin(torch.pi * x) * torch.cos(torch.pi * c * t)
            return _delta * torch.sin(_mode * torch.pi / _L * x) * torch.cos(_mode * torch.pi * c / _L * t)
        else: # NumPy
            x, t = xt[:, 0:1], xt[:, 1:2]
            c = np.sqrt(self.c_squared)
            if self.multi:
                return self.delta * np.sin(self.mode * np.pi * x) * np.cos(self.mode * np.pi * c * t) + np.sin(np.pi * x) * np.cos(np.pi * c * t)
            return self.delta * np.sin(self.mode * np.pi / self.L * x) * np.cos(self.mode * np.pi * c / self.L * t)
            
    def get_plot_amplitude(self):
        """The maximum amplitude for this standing wave is simply delta."""
        return self.delta
    
    def get_output_transform(self):
        # def output_transform(inputs, outputs):
        """
        Custom output transform for the wave equation.
        This is used to apply any specific transformations to the output of the model.
        For wave equations, we apply u(x, t) = b(x,t) + phi(x,t) * u_nn(x,t)
        where b(x,t) is the initial condition solution and phi(x,t) is the ADF. u_nn(x,t) is the neural network output.
        """
        def output_transform(inputs, v_raw_output):
            x, t = inputs[:, 0:1], inputs[:, 1:2]
            
            # Term for u(x,0) = f(x)
            # Uses torch.sin directly, just like bkd.sin in the standalone script
            initial_condition_term = self.delta * torch.sin(self.mode * np.pi / self.L * x[:, 0:1])
            
            # Factor for u(0,t)=u(L,t)=0
            boundary_factor = x * (self.L - x)
            
            # Factor for u_t(x,0)=0
            initial_velocity_factor = t**2
            
            u = initial_condition_term + boundary_factor * initial_velocity_factor * v_raw_output
            return u
        return output_transform
        #     x, t = inputs[:, 0:1], inputs[:, 1:2]
        #     b = self.get_initial_condition_func()(x)
        #     phi = x * (self.L - x) * t**2
        #     return b + phi * outputs
        # return output_transform
        