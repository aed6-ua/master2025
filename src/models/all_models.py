import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import deepxde as dde
import os
import io
from .base_model import BaseModel


def create_mlp(input_dim, output_dim, hidden_dim, num_layers, activation_fn=nn.ReLU):
    """Creates a simple MLP."""
    layers = [nn.Linear(input_dim, hidden_dim), activation_fn()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

# Map activation strings from config to torch functions
ACTIVATION_MAP = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'silu': nn.SiLU
}

class StandardPINN(BaseModel):
    """
    A standard Physics-Informed Neural Network (PINN).
    It uses a single Multi-Layer Perceptron (MLP) to approximate the solution u(x,t).
    This model is problem-agnostic.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Get MLP parameters from the config dictionary
        model_cfg = self.config.get('model', {})
        mlp_cfg = model_cfg.get('mlp', {})
        hidden_dim = mlp_cfg.get('hidden_dim', 30)
        num_layers = mlp_cfg.get('num_layers', 3)
        activation = mlp_cfg.get('activation', 'tanh')
        self.output_dim = model_cfg.get('output_dim', 1)
        
        self.fnn = dde.nn.FNN([2] + [hidden_dim] * num_layers + [self.output_dim], activation, "Glorot normal")
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized StandardPINN with {num_layers} hidden layers of {hidden_dim} neurons each.")

    def forward(self, x):
        # The forward pass is handled by the parent dde.nn.FNN class
        return self.fnn(x)
    
class BCNN(BaseModel):
    """
    A Boundary Constrained Physics-Informed Neural Network (PINN).
    It uses a single Multi-Layer Perceptron (MLP) to approximate the solution u(x,t) and applies boundary constraints.
    """
    def __init__(self, config, output_transform=None):
        super().__init__(config)
        
        # Get MLP parameters from the config dictionary
        model_cfg = self.config.get('model', {})
        mlp_cfg = model_cfg.get('mlp', {})
        hidden_dim = mlp_cfg.get('hidden_dim', 30)
        num_layers = mlp_cfg.get('num_layers', 3)
        activation = mlp_cfg.get('activation', 'tanh')
        self.output_dim = model_cfg.get('output_dim', 1)
        
        self.fnn = dde.nn.FNN([2] + [hidden_dim] * num_layers + [self.output_dim], activation, "Glorot normal")
        self.fnn.apply_output_transform(output_transform)
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized StandardPINN with {num_layers} hidden layers of {hidden_dim} neurons each.")

    def forward(self, x):
        # The forward pass is handled by the parent dde.nn.FNN class
        return self.fnn(x)
    


class SpectralNN(BaseModel):
    """
    A Spectral Neural Network with learnable spatial frequencies and time-dependent coefficients.
    
    The solution u(x,t) is approximated by:
       u(x,t) = sum [ A_n(t)*cos(k_n*x) + B_n(t)*sin(k_n*x) ]
    """

    def __init__(self, config):
        super().__init__(config)

        # --- Extract configuration ---
        model_cfg = self.config.get('model', {})
        mlp_cfg = model_cfg.get('mlp', {})
        self.n_freq_k_learnable = model_cfg.get('n_freq_k_learnable', 10)
        self.L = self.config.get('problem', {}).get('L', 1.0)
        self.T = self.config.get('problem', {}).get('T', 1.0)
        self.output_dim = model_cfg.get('output_dim', 1)
        
        # --- Parameters for the Learnable Frequencies ---
        # Initialize learnable spatial frequencies k_n
        k_init_bounds = model_cfg.get('k_init_bounds', (2.0, 40.0))
        initial_k_guess = torch.empty(self.n_freq_k_learnable, 1)
        nn.init.uniform_(initial_k_guess, a=k_init_bounds[0], b=k_init_bounds[1])
        self.k_learnable_params = nn.Parameter(initial_k_guess)
        self.k_max_estimate = k_init_bounds[1] # For scaling input to the MLP

        # MLP for the time-dependent coefficients A_n(t) and B_n(t)
        self.mlp_coeffs_learnable = create_mlp(
            input_dim=2, # (k_scaled, t_scaled)
            output_dim=2 * self.output_dim, # (A_n_u, B_n_u, A_n_v, B_n_v, ...)
            hidden_dim=mlp_cfg.get('hidden_dim', 30),
            num_layers=mlp_cfg.get('num_layers', 3),
            activation_fn=nn.Tanh
        )

        # Count total parameters
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized SpectralNN with {self.num_parameters} parameters.")
        
    def generate_k_n(self):
        """Applies softplus to ensure learned frequencies are positive."""
        return torch.nn.functional.softplus(self.k_learnable_params.T)

    def forward(self, inputs):
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        if self.output_dim == 0: return torch.zeros_like(x)

        N = x.shape[0]
        t_scaled = (2.0 * t / self.T) - 1.0

        k_learnable = self.generate_k_n()
        k_learnable_scaled = (2.0 * k_learnable / self.k_max_estimate) - 1.0

        # Expand for batch processing: Shape -> (N, n_freq_k_learnable, 1)
        t_expanded_learnable = t_scaled.expand(-1, self.n_freq_k_learnable).unsqueeze(-1)
        k_expanded_learnable = k_learnable_scaled.expand(N, -1).unsqueeze(-1)

        # Get coefficients A_n(t), B_n(t) from the "learnable" MLP
        mlp_input_learnable = torch.cat([k_expanded_learnable, t_expanded_learnable], dim=2)
        coeffs_learnable = self.mlp_coeffs_learnable(mlp_input_learnable.view(-1, 2)).view(N, self.n_freq_k_learnable, 2 * self.output_dim)

        kx_learnable = x @ k_learnable
        cos_kx_learnable = torch.cos(kx_learnable)
        sin_kx_learnable = torch.sin(kx_learnable)
        
        output_components = []
        for i in range(self.output_dim):
            A_n = coeffs_learnable[..., 2*i]
            B_n = coeffs_learnable[..., 2*i + 1]
            
            sum_cos = torch.sum(A_n * cos_kx_learnable, dim=1, keepdim=True)
            sum_sin = torch.sum(B_n * sin_kx_learnable, dim=1, keepdim=True)
            
            u_component = sum_cos + sum_sin
            output_components.append(u_component)

        return torch.cat(output_components, dim=1)

    def log_specific_params(self, mlflow):
        # Override the base method to log the final learned frequencies.
        print(f"Logging specific parameters for {self.__class__.__name__}")
        if mlflow.active_run():
            final_k_learnable = self.generate_k_n().detach().cpu().numpy().flatten()
            
            # Log as individual metrics
            # for i, k_val in enumerate(final_k_learnable):
            #     mlflow.log_metric(f"final_k_learnable_{i}", k_val)
            
            # Log as CSV artifact
            df_k = pd.DataFrame({
            'frequency_index': range(len(final_k_learnable)),
            'k_value': final_k_learnable
            })
            k_buffer = io.StringIO()
            df_k.to_csv(k_buffer, index=True)
            mlflow.log_text(k_buffer.getvalue(), artifact_file=f"model_parameters/final_learned_frequencies.csv")
            
            print(f"Logged final learned frequencies: {np.sort(final_k_learnable)}")
            print(f"Logged frequencies to artifact: model_parameters/final_learned_frequencies.csv")

        self.log_coefficient_maps(mlflow)

    def log_coefficient_maps(self, mlflow, k_samples=200, t_samples=100):
        """
        Generates and logs the full coefficient data (A and B) as a function of k and t.
        This should be called once at the end of training.
        """
        print("Generating and logging full A(k,t) and B(k,t) coefficient data...")
        self.eval() # Set model to evaluation mode

        with torch.no_grad():
            # 1. Create a grid of k and t values to sample the MLP
            k_n_final = self.generate_k_n()
            # Ensure the plot range covers all learned frequencies
            k_max_plot = torch.max(k_n_final) * 1.2 
            k_grid = torch.linspace(0, k_max_plot, k_samples, device=self.k_learnable_params.device)
            t_grid = torch.linspace(0, self.T, t_samples, device=self.k_learnable_params.device)
            
            # Scale for MLP input
            k_grid_scaled = (2.0 * k_grid / self.k_max_estimate) - 1.0
            t_grid_scaled = (2.0 * t_grid / self.T) - 1.0
            
            # Create a meshgrid of the scaled inputs for batch processing
            K_scaled, T_scaled = torch.meshgrid(k_grid_scaled, t_grid_scaled, indexing='ij')
            mlp_input_flat = torch.stack([K_scaled.flatten(), T_scaled.flatten()], dim=1)
            
            # 2. Get MLP output over the entire grid
            coeffs_flat = self.mlp_coeffs_learnable(mlp_input_flat)
            
            # Create the long-form DataFrame directly
            # This is much more flexible than saving separate matrices.
            df_coeffs = pd.DataFrame({
                # Un-scale k and t back to their original values for interpretability
                'k': K_scaled.flatten().cpu().numpy() * self.k_max_estimate / 2.0 + self.k_max_estimate / 2.0,
                't': T_scaled.flatten().cpu().numpy() * self.T / 2.0 + self.T / 2.0,
                'A': coeffs_flat[:, 0].cpu().numpy(),
                'B': coeffs_flat[:, 1].cpu().numpy()
            })
            
            # Add an amplitude column for convenience
            df_coeffs['Amplitude'] = np.sqrt(df_coeffs['A']**2 + df_coeffs['B']**2)

            # 3. Save the comprehensive DataFrame as a single artifact
            coeffs_buffer = io.StringIO()
            df_coeffs.to_csv(coeffs_buffer, index=True)
            mlflow.log_text(coeffs_buffer.getvalue(), artifact_file=f"coefficient_data/full_coefficient_map.csv")
            print(f"Logged full coefficient map to artifact: coefficient_data/full_coefficient_map.csv")

            try:
                # Log the specific coeff values for each of the final learned frequencies in a separate dataframe
                k_n_final_scaled = (2.0 * k_n_final / self.k_max_estimate) - 1.0
                # Meshgrid for the final k_n values
                k_n_final_grid = k_n_final_scaled.expand(1, -1)
                t_grid_scaled_expanded = t_grid_scaled.expand(-1, self.n_freq_k_learnable).unsqueeze(-1)
                mlp_input_final = torch.cat([k_n_final_grid, t_grid_scaled_expanded], dim=2)
                coeffs_final = self.mlp_coeffs_learnable(mlp_input_final.view(-1, 2)).view(1, self.n_freq_k_learnable, 2)
                df_coeffs_final = pd.DataFrame({
                    'k': k_n_final_scaled.cpu().numpy(),
                    't': t_grid_scaled.cpu().numpy(),
                    'A': coeffs_final[:, 0].cpu().numpy(),
                    'B': coeffs_final[:, 1].cpu().numpy()
                })
                buffer = io.StringIO()
                df_coeffs_final.to_csv(buffer, index=False)
                mlflow.log_text(buffer.getvalue(), artifact_file=f"coefficient_data/final_coefficients.csv")
                print(f"Logged final coefficients for learned frequencies to artifact: coefficient_data/final_coefficients.csv")
            except Exception as e:
                print(f"Error logging final coefficients: {e}")
        


            # Log some visualizations
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Coefficient heatmap Spectrum Plot for A values and B values
            fig = make_subplots(rows=1, cols=2, subplot_titles=("A Coefficients", "B Coefficients"))
            
            fig.add_trace(
                go.Heatmap(
                    z=df_coeffs.pivot(index="k", columns="t", values="A").values,
                    x=df_coeffs['t'].unique(),
                    y=df_coeffs['k'].unique(),
                    colorscale='Viridis',
                    colorbar=dict(title='A Coefficients')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Heatmap(
                    z=df_coeffs.pivot(index="k", columns="t", values="B").values,
                    x=df_coeffs['t'].unique(),
                    y=df_coeffs['k'].unique(),
                    colorscale='Viridis',
                    colorbar=dict(title='B Coefficients')
                ),
                row=1, col=2
            )

            # Show in the heatmaps where the final learned frequencies k_n are
            for k_val in k_n_final.detach().cpu().numpy().flatten():
                fig.add_shape(
                    type="line",
                    x0=0, x1=self.T,
                    y0=k_val.item(), y1=k_val.item(),
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=0, x1=self.T,
                    y0=k_val.item(), y1=k_val.item(),
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=2
                )
            fig.update_layout(
                title_text=f"Coefficient Spectrum for {self.__class__.__name__}",
                width=1000, height=500,
                xaxis_title="Time (t)",
                yaxis_title="Frequency (k)"
            )
            coeffs_fig_path = "coefficient_spectrum_plot.html"
            # fig.write_html(coeffs_fig_path)
            # mlflow.log_artifact(coeffs_fig_path, "coefficient_data")
            mlflow.log_figure(fig, f'coefficient_data/{coeffs_fig_path}')
            print(f"Logged coefficient spectrum plot to artifact: coefficient_data/{coeffs_fig_path}")

        self.train() # Set back to training mode



class BCSNN(BaseModel):
    """
    A Spectral Neural Network with fixed + learnable spatial frequencies and time-dependent coefficients.
    
    The solution u(x,t) is approximated by:
    u(x,t) = phi_x(x) * sum [ epsilon(t) * u_0(k_m) + phi_t(t) * u(k_m, t) ]
    """

    def __init__(self, config):
        super().__init__(config)

        # --- Extract configuration ---
        model_cfg = self.config.get('model', {})
        mlp_cfg = model_cfg.get('mlp', {})
        self.n_freq = model_cfg.get('n_freq_fixed', 10)
        self.L = self.config.get('problem', {}).get('L', 1.0)
        self.T = self.config.get('problem', {}).get('T', 1.0)

        # --- Initialize the learnable parameter 'p' ---
        # We want p to start at a reasonable value, e.g., 0.1.
        # We use the inverse sigmoid (logit) to find the initial raw value.
        # logit(y) = log(y / (1-y))
        initial_p = model_cfg.get('phi_x_p_initial', 0.1)
        p_raw_init_val = np.log(initial_p / (1 - initial_p))
        
        # The raw, unconstrained parameter that the optimizer will update
        p_raw_init = torch.tensor(p_raw_init_val, dtype=torch.float64)
        self.p_raw = nn.Parameter(p_raw_init, requires_grad=True)

        # MLP for the time-dependent coefficients A_n(t) and B_n(t)
        self.mlp_coeffs_learnable = create_mlp(
            input_dim=2, # (k_scaled, t_scaled)
            output_dim=2, # (A_n, B_n)
            hidden_dim=mlp_cfg.get('hidden_dim', 30),
            num_layers=mlp_cfg.get('num_layers', 3),
            activation_fn=nn.Tanh
        )
        self.device = self.mlp_coeffs_learnable[0].weight.device

        # Random init the alpha parameter
        alpha_init = torch.tensor(np.random.uniform(100.0, 200.0), dtype=torch.float64).unsqueeze(0)
        self.alpha = nn.Parameter(alpha_init, requires_grad=True)
        self.device = self.alpha.device

        self.to(self.device)

        # Count total parameters
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized BCSNN with {self.num_parameters} parameters.")

    def load_ic_data(self, k_m_values, a_hat_values, b_hat_values):
        """
        Loads the Fourier decomposition of the initial condition.
        This data is used for the main (non-learnable) part of the solution.
        """
        print("BCSNN: Loading initial condition Fourier data...")
        
        self.n_freq_k_fixed = len(k_m_values)
        if self.n_freq_k_fixed == 0:
            print("Warning: No fixed frequencies provided from IC. Main term will be zero.")
            return

        self.k_fixed = torch.tensor(k_m_values, dtype=torch.float64, device=self.device).unsqueeze(0)
        self.a_hat_ic = torch.tensor(a_hat_values, dtype=torch.float64, device=self.device).unsqueeze(0)
        self.b_hat_ic = torch.tensor(b_hat_values, dtype=torch.float64, device=self.device).unsqueeze(0)
        self.k_max = torch.max(self.k_fixed).item()
        print(f"Loaded {self.n_freq_k_fixed} fixed frequencies for the main term.")
        
    def generate_k_n(self):
        return self.k_fixed

    def forward(self, inputs):
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        N = x.shape[0]
        t_scaled = (2.0 * t / self.T) - 1.0


        # --- Epsilon Function ---
        # This function is 1 at t=0 and decays to 0. It "turns on" the IC coefficients.
        epsilon_t = torch.exp(-t**2 * self.alpha)

        # --- Distance Function (phi) ---
        # This function is 0 at t=0 and at x=0, L.
        # It ensures the correction term and the t>0 part of the main term
        # do not interfere with the hard-enforced IC and BCs.
        phi_t = (1 - epsilon_t)

        # --- Spatial Function with learnable 'p' ---
        # 1. Transform the raw parameter to the constrained range (0, 1)
        p = torch.sigmoid(self.p_raw)

        # 2. Create the base function
        phi_base = (4.0 / (self.L**2)) * x * (self.L - x)
        
        # 3. Apply the learnable power 'p'
        phi_x = torch.pow(phi_base, p)

        # Part 1: Initial Condition Term
        k_n = self.generate_k_n()
        kx = x @ k_n
        cos_kx = torch.cos(kx)
        sin_kx = torch.sin(kx)
        
        u_0 = torch.sum(self.a_hat_ic * cos_kx, dim=1, keepdim=True) + \
              torch.sum(self.b_hat_ic * sin_kx, dim=1, keepdim=True)
              
        u_ic_term = u_0 * epsilon_t

        # Part 2: Network Correction Term
        k_scaled = (2.0 * k_n / self.k_max) - 1.0
        t_scaled = (2.0 * t / self.T) - 1.0

        t_expanded = t_scaled.expand(-1, self.n_freq_k_fixed).unsqueeze(-1)
        k_expanded = k_scaled.expand(N, -1).unsqueeze(-1)

        mlp_input = torch.cat([k_expanded, t_expanded], dim=2)
        coeffs_nn = self.mlp_coeffs_learnable(mlp_input.view(-1, 2)).view(N, self.n_freq_k_fixed, 2)
        A_nn = coeffs_nn[..., 0]
        B_nn = coeffs_nn[..., 1]
        
        fourier_series_nn = torch.sum(A_nn * cos_kx, dim=1, keepdim=True) + \
                            torch.sum(B_nn * sin_kx, dim=1, keepdim=True)

        u_nn_term = phi_x * phi_t * fourier_series_nn
        
        u_final = u_ic_term + u_nn_term
        
        return u_final

    def log_specific_params(self, mlflow):
        # Override the base method to log the final learned frequencies.
        print(f"Logging specific parameters for {self.__class__.__name__}")
        if mlflow.active_run():
            final_k_learnable = self.generate_k_n().detach().cpu().numpy().flatten()
            
            # Log as individual metrics
            # for i, k_val in enumerate(final_k_learnable):
            #     mlflow.log_metric(f"final_k_learnable_{i}", k_val)
            
            # Log as CSV artifact
            df_k = pd.DataFrame({
            'frequency_index': range(len(final_k_learnable)),
            'k_value': final_k_learnable
            })
            k_buffer = io.StringIO()
            df_k.to_csv(k_buffer, index=True)
            mlflow.log_text(k_buffer.getvalue(), artifact_file=f"model_parameters/final_learned_frequencies.csv")
            
            print(f"Logged final learned frequencies: {np.sort(final_k_learnable)}")
            print(f"Logged frequencies to artifact: model_parameters/final_learned_frequencies.csv")

            # Log the alpha parameter
            mlflow.log_metric("alpha", self.alpha.detach().cpu().numpy().item())

            # Log the raw p parameter
            p_value = torch.sigmoid(self.p_raw).detach().cpu().numpy().item()
            mlflow.log_metric("p_value", p_value)

        self.log_coefficient_maps(mlflow)

    def log_coefficient_maps(self, mlflow, k_samples=200, t_samples=100):
        """
        Generates and logs the full coefficient data (A and B) as a function of k and t.
        This should be called once at the end of training.
        """
        print("Generating and logging full A(k,t) and B(k,t) coefficient data...")
        self.eval() # Set model to evaluation mode

        with torch.no_grad():
            # 1. Create a grid of k and t values to sample the MLP
            k_n_final = self.generate_k_n()
            # Ensure the plot range covers all learned frequencies
            k_max_plot = torch.max(k_n_final) * 1.2 
            k_grid = torch.linspace(0, k_max_plot, k_samples, device=self.device)
            t_grid = torch.linspace(0, self.T, t_samples, device=self.device)
            
            # Scale for MLP input
            k_grid_scaled = (2.0 * k_grid / self.k_max) - 1.0
            t_grid_scaled = (2.0 * t_grid / self.T) - 1.0
            
            # Create a meshgrid of the scaled inputs for batch processing
            K_scaled, T_scaled = torch.meshgrid(k_grid_scaled, t_grid_scaled, indexing='ij')
            mlp_input_flat = torch.stack([K_scaled.flatten(), T_scaled.flatten()], dim=1)
            
            # 2. Get MLP output over the entire grid
            coeffs_flat = self.mlp_coeffs_learnable(mlp_input_flat)
            
            # Create the long-form DataFrame directly
            # This is much more flexible than saving separate matrices.
            df_coeffs = pd.DataFrame({
                # Un-scale k and t back to their original values for interpretability
                'k': K_scaled.flatten().cpu().numpy() * self.k_max / 2.0 + self.k_max / 2.0,
                't': T_scaled.flatten().cpu().numpy() * self.T / 2.0 + self.T / 2.0,
                'A': coeffs_flat[:, 0].cpu().numpy(),
                'B': coeffs_flat[:, 1].cpu().numpy()
            })
            
            # Add an amplitude column for convenience
            df_coeffs['Amplitude'] = np.sqrt(df_coeffs['A']**2 + df_coeffs['B']**2)

            # 3. Save the comprehensive DataFrame as a single artifact
            coeffs_buffer = io.StringIO()
            df_coeffs.to_csv(coeffs_buffer, index=True)
            mlflow.log_text(coeffs_buffer.getvalue(), artifact_file=f"coefficient_data/full_coefficient_map.csv")
            print(f"Logged full coefficient map to artifact: coefficient_data/full_coefficient_map.csv")

            try:
                # Log the final learned frequencies coefficients
                k_n_final_scaled = (2.0 * k_n_final / self.k_max) - 1.0
                # Meshgrid for the final k_n values
                k_n_final_grid = k_n_final_scaled.expand(1, -1)
                t_grid_scaled_expanded = t_grid_scaled.expand(-1, self.n_freq_k_fixed).unsqueeze(-1)
                mlp_input_final = torch.cat([k_n_final_grid, t_grid_scaled_expanded], dim=2)
                coeffs_final = self.mlp_coeffs_learnable(mlp_input_final.view(-1, 2)).view(1, self.n_freq_k_fixed, 2)
                df_coeffs_final = pd.DataFrame({
                    'k': k_n_final_scaled.cpu().numpy(),
                    't': t_grid_scaled.cpu().numpy(),
                    'A': coeffs_final[:, 0].cpu().numpy(),
                    'B': coeffs_final[:, 1].cpu().numpy()
                })
                buffer = io.StringIO()
                df_coeffs_final.to_csv(buffer, index=False)
                mlflow.log_text(buffer.getvalue(), artifact_file=f"coefficient_data/final_coefficients.csv")
                print(f"Logged final coefficients for learned frequencies to artifact: coefficient_data/final_coefficients.csv")
            except Exception as e:
                print(f"Error logging final coefficients: {e}")

            # Log some visualizations
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Coefficient heatmap Spectrum Plot for A values and B values
            fig = make_subplots(rows=1, cols=2, subplot_titles=("A Coefficients", "B Coefficients"))
            
            fig.add_trace(
                go.Heatmap(
                    z=df_coeffs.pivot(index="k", columns="t", values="A").values,
                    x=df_coeffs['t'].unique(),
                    y=df_coeffs['k'].unique(),
                    colorscale='Viridis',
                    colorbar=dict(title='A Coefficients')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Heatmap(
                    z=df_coeffs.pivot(index="k", columns="t", values="B").values,
                    x=df_coeffs['t'].unique(),
                    y=df_coeffs['k'].unique(),
                    colorscale='Viridis',
                    colorbar=dict(title='B Coefficients')
                ),
                row=1, col=2
            )

            # Show in the heatmaps where the final learned frequencies k_n are
            for k_val in k_n_final.detach().cpu().numpy().flatten():
                fig.add_shape(
                    type="line",
                    x0=0, x1=self.T,
                    y0=k_val.item(), y1=k_val.item(),
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=0, x1=self.T,
                    y0=k_val.item(), y1=k_val.item(),
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=2
                )
            fig.update_layout(
                title_text=f"Coefficient Spectrum for {self.__class__.__name__}",
                width=1000, height=500,
                xaxis_title="Time (t)",
                yaxis_title="Frequency (k)"
            )
            coeffs_fig_path = "coefficient_spectrum_plot.html"
            # fig.write_html(coeffs_fig_path)
            # mlflow.log_artifact(coeffs_fig_path, "coefficient_data")
            mlflow.log_figure(fig, f'coefficient_data/{coeffs_fig_path}')
            print(f"Logged coefficient spectrum plot to artifact: coefficient_data/{coeffs_fig_path}")

        self.train() # Set back to training mode


class VarBCSNN(BCSNN):
    """
    A Boundary Constrained Spectral Neural Network with a hybrid frequency basis.
    - Dominant frequencies from the IC are kept FIXED.
    - Secondary frequencies from the IC are initialized as LEARNABLE parameters.
    """

    def __init__(self, config):
        # Call BaseModel's init
        super(BCSNN, self).__init__(config) 

        # --- Extract configuration ---
        model_cfg = self.config.get('model', {})
        mlp_cfg = model_cfg.get('mlp', {})
        self.n_freq_fixed = model_cfg.get('n_freq_fixed')
        self.n_freq_learnable = model_cfg.get('n_freq_learnable')
        self.L = self.config.get('problem', {}).get('L', 1.0)
        self.T = self.config.get('problem', {}).get('T', 1.0)

        # --- Initialize the learnable parameter 'p' ---
        # We want p to start at a reasonable value, e.g., 0.1.
        # We use the inverse sigmoid (logit) to find the initial raw value.
        # logit(y) = log(y / (1-y))
        initial_p = model_cfg.get('phi_x_p_initial', 0.1)
        p_raw_init_val = np.log(initial_p / (1 - initial_p))
        
        # The raw, unconstrained parameter that the optimizer will update
        p_raw_init = torch.tensor(p_raw_init_val, dtype=torch.float64)
        self.p_raw = nn.Parameter(p_raw_init, requires_grad=True)
        
        # --- MLP ---
        self.mlp_coeffs_learnable = create_mlp(
            input_dim=2, output_dim=2,
            hidden_dim=mlp_cfg.get('hidden_dim', 30),
            num_layers=mlp_cfg.get('num_layers', 3),
            activation_fn=nn.Tanh
        )
        
        # --- Learnable Parameters ---
        # Initialize learnable spatial frequencies k'_n RANDOMLY
        k_init_bounds = model_cfg.get('k_init_bounds', (2.0, 40.0))
        initial_k_guess = torch.empty(self.n_freq_learnable, 1, dtype=torch.float64)
        nn.init.uniform_(initial_k_guess, a=k_init_bounds[0], b=k_init_bounds[1])
        self.k_learnable_params = nn.Parameter(initial_k_guess, requires_grad=True)
        self.k_max_estimate = k_init_bounds[1] # For scaling input to the MLP

        # Learnable decay parameter
        alpha_init = torch.tensor(np.random.uniform(0.1, 100.0), dtype=torch.float64)
        self.alpha = nn.Parameter(alpha_init, requires_grad=True)
        self.device = self.alpha.device
        

        # Buffers for all data
        self.register_buffer('k_fixed', torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer('a_hat_ic_fixed', torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer('b_hat_ic_fixed', torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer('a_hat_ic_learnable', torch.zeros(1, self.n_freq_learnable, dtype=torch.float64))
        self.register_buffer('b_hat_ic_learnable', torch.zeros(1, self.n_freq_learnable, dtype=torch.float64))
        self.k_max = 1.0

        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized VarBCSNN with {self.num_parameters} parameters.")

    def load_ic_data(self, k_dominant, a_dominant, b_dominant):
        """
        Loads the Fourier decomposition of the DOMINANT modes of the IC.
        The learnable frequencies are initialized independently and randomly.
        """
        print("VarBCSNN: Loading DOMINANT initial condition Fourier data...")
        device = self.alpha.device
        
        if len(k_dominant) != self.n_freq_fixed:
             print(f"Warning: Received {len(k_dominant)} dominant modes, but expected {self.n_freq_fixed}.")

        # Load Fixed (Dominant) Data
        if len(k_dominant) > 0:
            self.k_fixed.data = torch.tensor(k_dominant, dtype=torch.float64, device=device).unsqueeze(0)
            self.a_hat_ic_fixed.data = torch.tensor(a_dominant, dtype=torch.float64, device=device).unsqueeze(0)
            self.b_hat_ic_fixed.data = torch.tensor(b_dominant, dtype=torch.float64, device=device).unsqueeze(0)
            print(f"Loaded {len(k_dominant)} fixed dominant frequencies.")
        else:
            print("No fixed frequencies provided.")
        
        # Update the max k value for scaling, now just based on fixed freqs and the random init range
        k_fixed_max = torch.max(self.k_fixed).item() if self.k_fixed.numel() > 0 else 0.0
        self.k_max = max(k_fixed_max, self.k_max_estimate)

    # The `generate_k_n` and `forward` methods can remain identical to the previous version,
    # as they already correctly handle the combination of `k_fixed` and `k_learnable_params`.
    def generate_k_n(self):
        """Combines fixed and learnable frequencies into a single tensor for the forward pass."""
        k_learnable = torch.nn.functional.softplus(self.k_learnable_params.T)
        return torch.cat([self.k_fixed, k_learnable], dim=1)

    def forward(self, inputs):
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        N = x.shape[0]
        
        a_hat_ic = torch.cat([self.a_hat_ic_fixed, self.a_hat_ic_learnable], dim=1)
        b_hat_ic = torch.cat([self.b_hat_ic_fixed, self.b_hat_ic_learnable], dim=1)
        
        k_n = self.generate_k_n()
        n_total_freqs = k_n.shape[1]

        t_scaled = (2.0 * t / self.T) - 1.0
        k_scaled = (2.0 * k_n / self.k_max) - 1.0

        t_expanded = t_scaled.expand(-1, n_total_freqs).unsqueeze(-1)
        k_expanded = k_scaled.expand(N, -1).unsqueeze(-1)
        
        epsilon = torch.exp(-t**2 * self.alpha)
        phi_t = (1 - epsilon)

        # --- Spatial Function with learnable 'p' ---
        # 1. Transform the raw parameter to the constrained range (0, 1)
        p = torch.sigmoid(self.p_raw)

        # 2. Create the base function
        phi_base = (4.0 / (self.L**2)) * x * (self.L - x)
        
        # 3. Apply the learnable power 'p'
        phi_x = torch.pow(phi_base, p)


        kx = x @ k_n
        cos_kx = torch.cos(kx)
        sin_kx = torch.sin(kx)

        u_0 = torch.sum(a_hat_ic * cos_kx, dim=1, keepdim=True) + \
              torch.sum(b_hat_ic * sin_kx, dim=1, keepdim=True)
              
        u_ic_term = u_0 * epsilon
        


        mlp_input = torch.cat([k_expanded, t_expanded], dim=2)
        coeffs_nn = self.mlp_coeffs_learnable(mlp_input.view(-1, 2)).view(N, n_total_freqs, 2)
        A_nn = coeffs_nn[..., 0]
        B_nn = coeffs_nn[..., 1]
        
        fourier_series_nn = torch.sum(A_nn * cos_kx, dim=1, keepdim=True) + \
                            torch.sum(B_nn * sin_kx, dim=1, keepdim=True)

        u_nn_term = phi_x * phi_t * fourier_series_nn
        
        u_final = u_ic_term + u_nn_term
        
        return u_final
    
