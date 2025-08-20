import numpy as np
import torch
import deepxde as dde
from .base_problem import BaseProblem

def ic_ramp(x_coords, L_domain):
    """
    Creates a ramp initial condition.
    Value increases linearly from 0 to a maximum at L.
    x_coords: NumPy array of shape (N, 1) representing x coordinates.
    L_domain: Length of the domain.
    """
    u0 = np.zeros_like(x_coords[:, 0])
    ramp_height = 5.0  # Arbitrary height
    u0[:, 0] = ramp_height * (x_coords[:, 0] / L_domain)
    return u0

def ic_step_function(x_coords, L_domain):
    """
    Creates a step function initial condition.
    Value is high in the middle third, low otherwise.
    x_coords: NumPy array of shape (N, 1) representing x coordinates.
    L_domain: Length of the domain.
    """
    u0 = np.zeros_like(x_coords[:, 0])
    center_start = L_domain / 3.0
    center_end = 2 * L_domain / 3.0
    step_height = 5.0  # Arbitrary height
    
    condition = (x_coords[:, 0] >= center_start) & (x_coords[:, 0] <= center_end)
    u0[condition] = step_height
    return u0

def ic_triangular_pulse(x_coords, L_domain):
    """
    Creates a triangular pulse initial condition centered in the domain.
    x_coords: NumPy array of shape (N, 1) representing x coordinates.
    L_domain: Length of the domain.
    """
    u0 = np.zeros_like(x_coords[:, 0])
    peak_position = L_domain / 2.0
    pulse_width_half = L_domain / 4.0 # Half-width of the triangle base
    peak_height = 4.0 # Arbitrary height

    # Left slope
    mask_left = (x_coords[:, 0] >= peak_position - pulse_width_half) & (x_coords[:, 0] < peak_position)
    u0[mask_left] = peak_height * (1 - (peak_position - x_coords[mask_left, 0]) / pulse_width_half)

    # Right slope
    mask_right = (x_coords[:, 0] >= peak_position) & (x_coords[:, 0] <= peak_position + pulse_width_half)
    u0[mask_right] = peak_height * (1 - (x_coords[mask_right, 0] - peak_position) / pulse_width_half)
    
    return u0

def ic_piecewise_linear(x_coords, L_domain):
    """
    Creates a more complex piecewise linear initial condition.
    Example: A "W" shape or an asymmetric shape.
    x_coords: NumPy array of shape (N, 1) representing x coordinates.
    L_domain: Length of the domain.
    """
    u0 = np.zeros_like(x_coords[:, 0])
    x = x_coords[:, 0]

    # Define points (x_val, y_val) for the piecewise linear function
    # These points must be ordered by x_val
    # An "M" shape
    points = [
        (0.0 * L_domain, 0.0),
        (0.2 * L_domain, 3.0),
        (0.4 * L_domain, 1.0),
        (0.6 * L_domain, 4.0),
        (0.8 * L_domain, 0.5),
        (1.0 * L_domain, 2.0)
    ]

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]

        # Mask for the current segment
        if i == len(points) - 2: # Last segment, include endpoint
            mask = (x >= x1) & (x <= x2)
        else: # Other segments, exclude endpoint to avoid overlap with next segment's start
            mask = (x >= x1) & (x < x2)
        
        if x2 == x1:
            if y1 != y2: print(f"Warning: Vertical line segment in piecewise IC at x={x1}")
            u0[mask] = y1 
        else:
            slope = (y2 - y1) / (x2 - x1)
            u0[mask] = y1 + slope * (x[mask] - x1)
            
    return u0

class HeatEquationProblem(BaseProblem):
    def __init__(self, config):
        problem_cfg = config['problem']
        self.L = problem_cfg['L']
        self.T = problem_cfg['T']
        self.alpha = problem_cfg['alpha']  # Diffusion coefficient
        self.ic_type = problem_cfg['ic_type']  # Type of initial condition
        self.ic_params = problem_cfg.get('ic_params', {})
        self.ic_bcs = problem_cfg.get('ic_bcs', {})
        super().__init__(config)
        self.plot_amplitude = self.get_plot_amplitude()

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
        du_t = dde.grad.jacobian(u, inputs, i=0, j=1)
        du_xx = dde.grad.hessian(u, inputs, i=0, j=0)
        return du_t - self.alpha * du_xx

    def get_ics_bcs(self):
        ic_bcs = []
        # 1. Boundary Conditions: u(0, t) = 0 and u(L, t) = 0
        if self.ic_bcs.get('bc', True):
            bc = dde.icbc.DirichletBC(
                self.geomtime, 
                lambda x: 0.0, 
                lambda x, on_boundary: on_boundary
            )
            ic_bcs.append(bc)
        # 2. Initial Condition for temperature: u(x, 0) = f(x)
        if self.ic_bcs.get('ic', True):
            ic = dde.icbc.IC(
                self.geomtime,
                self.get_initial_condition_func(),
                lambda _, on_initial: on_initial
            )
            ic_bcs.append(ic)
        return ic_bcs

    def get_initial_condition_func(self):
        def ic_func(x_input_np):
            if self.ic_type == 'sum_sine':
                u0_vals = np.zeros_like(x_input_np[:, 0])
                components = self.ic_params.get('sum_sine_components', [])
                for amp, k in components:
                    u0_vals += amp * np.sin(k * x_input_np[:, 0])
                return u0_vals.reshape(-1, 1)
            elif self.ic_type == 'step':
                return ic_step_function(x_input_np, self.L)
            elif self.ic_type == 'gaussian':
                amp = self.ic_params.get('gauss_amplitude', 5.0)
                center = self.ic_params.get('gauss_center', self.L/3)
                std_dev = self.ic_params.get('gauss_std_dev', self.L/50)
                return amp * np.exp(-((x_input_np[:, 0] - center)**2) / (2 * std_dev**2)).reshape(-1, 1)
            else:
                raise ValueError(f"Unknown IC Type: {self.ic_type}")
        return ic_func

    def analytical_solution(self, xt):
        is_torch = isinstance(xt, torch.Tensor)
        if is_torch:
            x, t = xt[:, 0:1], xt[:, 1:2]
            u_sol = torch.zeros_like(x)
            _L, _alpha = torch.tensor(self.L, device=xt.device), torch.tensor(self.alpha, device=xt.device)
        else: # NumPy
            x, t = xt[:, 0:1], xt[:, 1:2]
            u_sol = np.zeros_like(x)
            _L, _alpha = self.L, self.alpha

        if self.ic_type == 'sum_sine':
            for amplitude, k_val_ic in self.ic_params.get('sum_sine_components', []):
                if is_torch:
                    amp_t = torch.tensor(amplitude, device=xt.device, dtype=x.dtype)
                    k_t = torch.tensor(k_val_ic, device=xt.device, dtype=x.dtype)
                    lambda_n = _alpha * (k_t**2)
                    term = amp_t * torch.sin(k_t * x) * torch.exp(-lambda_n * t)
                else:
                    lambda_n = _alpha * (k_val_ic**2)
                    term = amplitude * np.sin(k_val_ic * x) * np.exp(-lambda_n * t)
                u_sol += term

        return u_sol
    
    def get_plot_amplitude(self):
        # Logic to determine the y-axis range for plots
        if self.ic_type == 'sum_sine':
            return sum(abs(amp) for amp, _ in self.ic_params.get('sum_sine_components', []))
        elif self.ic_type == 'gaussian':
            return self.ic_params.get('gauss_amplitude', 5.0)
        return 5.0 # default

    def get_output_transform(self):
        # def output_transform(inputs, outputs):
        """
        Custom output transform for the wave equation.
        This is used to apply any specific transformations to the output of the model.
        For wave equations, we apply u(x, t) = b(x,t) + phi(x,t) * u_nn(x,t)
        where b(x,t) is the initial condition solution and phi(x,t) is the ADF. u_nn(x,t) is the neural network output.
        """
        def output_transform(inputs, outputs):
            x, t = inputs[:, 0:1], inputs[:, 1:2]
            # convert x grad tensor to numpy if needed (use tensor.detach() for torch)
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            b = self.get_initial_condition_func()(x_np)
            # convert back to torch if needed
            if isinstance(b, np.ndarray):
                b = torch.tensor(b, device=inputs.device, dtype=inputs.dtype)

            exp = torch.exp(-50  * t)
            phi = x * (self.L - x) * (1-exp)
            return b * exp + phi * outputs
        return output_transform