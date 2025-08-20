import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import deepxde as dde
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
        
        self.fnn = dde.nn.FNN([2] + [hidden_dim] * num_layers + [1], activation, "Glorot normal")
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
        
        self.fnn = dde.nn.FNN([2] + [hidden_dim] * num_layers + [1], activation, "Glorot normal")
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
            output_dim=2, # (A_n, B_n)
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
        N = x.shape[0]
        t_scaled = (2.0 * t / self.T) - 1.0

        k_learnable = self.generate_k_n()
        k_learnable_scaled = (2.0 * k_learnable / self.k_max_estimate) - 1.0

        # Expand for batch processing: Shape -> (N, n_freq_k_learnable, 1)
        t_expanded_learnable = t_scaled.expand(-1, self.n_freq_k_learnable).unsqueeze(-1)
        k_expanded_learnable = k_learnable_scaled.expand(N, -1).unsqueeze(-1)

        # Get coefficients A_n(t), B_n(t) from the "learnable" MLP
        mlp_input_learnable = torch.cat([k_expanded_learnable, t_expanded_learnable], dim=2)
        coeffs_learnable = self.mlp_coeffs_learnable(mlp_input_learnable.view(-1, 2)).view(N, self.n_freq_k_learnable, 2)
        
        A_n = coeffs_learnable[..., 0]
        B_n = coeffs_learnable[..., 1]

        kx_learnable = x @ k_learnable
        cos_kx_learnable = torch.cos(kx_learnable)
        sin_kx_learnable = torch.sin(kx_learnable)
        
        sum_cos = torch.sum(A_n * cos_kx_learnable, dim=1, keepdim=True)
        sum_sin = torch.sum(B_n * sin_kx_learnable, dim=1, keepdim=True)
        
        u_final = sum_cos + sum_sin
        
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
            k_path = "final_learned_frequencies.csv"
            df_k.to_csv(k_path, index=False)
            mlflow.log_artifact(k_path, "model_parameters")
            
            print(f"Logged final learned frequencies: {np.sort(final_k_learnable)}")
            print(f"Logged frequencies to artifact: model_parameters/{k_path}")

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
            coeffs_path = "full_coefficient_map.csv"
            df_coeffs.to_csv(coeffs_path, index=False)
            mlflow.log_artifact(coeffs_path, "coefficient_data")
            print(f"Logged full coefficient map to artifact: coefficient_data/{coeffs_path}")

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
            fig.write_html(coeffs_fig_path)
            mlflow.log_artifact(coeffs_fig_path, "coefficient_data")
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
        self.n_freq_k_fixed = model_cfg.get('n_freq_k_fixed', 10)
        self.n_freq_k_learnable = model_cfg.get('n_freq_k_learnable', 10)
        self.L = self.config.get('problem', {}).get('L', 1.0)
        self.T = self.config.get('problem', {}).get('T', 1.0)
        
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
            output_dim=2, # (A_n, B_n)
            hidden_dim=mlp_cfg.get('hidden_dim', 30),
            num_layers=mlp_cfg.get('num_layers', 3),
            activation_fn=nn.Tanh
        )

        # Count total parameters
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized BCSNN with {self.num_parameters} parameters.")

    def load_ic_data(self, k_m_values, a_hat_values, b_hat_values):
        """
        Loads the Fourier decomposition of the initial condition.
        This data is used for the main (non-learnable) part of the solution.
        """
        print("BCSNN: Loading initial condition Fourier data...")
        device = self.k_learnable_params.device
        
        self.n_freq_k_fixed = len(k_m_values)
        if self.n_freq_k_fixed == 0:
            print("Warning: No fixed frequencies provided from IC. Main term will be zero.")
            return

        self.k_fixed = torch.tensor(k_m_values, dtype=torch.float64, device=device).unsqueeze(0)
        self.a_hat_ic = torch.tensor(a_hat_values, dtype=torch.float64, device=device).unsqueeze(0)
        self.b_hat_ic = torch.tensor(b_hat_values, dtype=torch.float64, device=device).unsqueeze(0)
        self.k_fixed_max = torch.max(self.k_fixed).item()
        print(f"Loaded {self.n_freq_k_fixed} fixed frequencies for the main term.")
        
    # def generate_k_n(self):
    #     """Applies softplus to ensure learned frequencies are positive."""
    #     return torch.nn.functional.softplus(self.k_learnable_params.T)

    def forward(self, inputs):
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        N = x.shape[0]
        t_scaled = (2.0 * t / self.T) - 1.0

        # print(f"x type: {x.dtype}, t type: {t.dtype}, N: {N}")

        # --- Epsilon Function ---
        # This function is 1 at t=0 and decays to 0. It "turns on" the IC coefficients.
        epsilon = torch.exp(-t)

        # --- Distance Function (phi) ---
        # This function is 0 at t=0 and at x=0, L.
        # It ensures the correction term and the t>0 part of the main term
        # do not interfere with the hard-enforced IC and BCs.
        phi_t = (1 - epsilon)

        k_fixed_scaled = (2.0 * self.k_fixed / self.k_max_estimate) - 1.0

        # Expand for batch processing: Shape -> (N, n_freq_k_fixed, 1)
        t_expanded_fixed = t_scaled.expand(-1, self.n_freq_k_fixed).unsqueeze(-1)
        k_expanded_fixed = k_fixed_scaled.expand(N, -1).unsqueeze(-1)

        # Get coefficients A_n(t), B_n(t) from the MLP
        mlp_input_fixed = torch.cat([k_expanded_fixed, t_expanded_fixed], dim=2)
        coeffs_fixed = self.mlp_coeffs_learnable(mlp_input_fixed.view(-1, 2)).view(N, self.n_freq_k_fixed, 2)
        
        A_n = coeffs_fixed[..., 0] * phi_t + epsilon * self.a_hat_ic
        B_n = coeffs_fixed[..., 1] * phi_t + epsilon * self.b_hat_ic

        kx_fixed = x @ self.k_fixed
        cos_kx_fixed = torch.cos(kx_fixed)
        sin_kx_fixed = torch.sin(kx_fixed)
        
        sum_cos = torch.sum(A_n * cos_kx_fixed, dim=1, keepdim=True)
        sum_sin = torch.sum(B_n * sin_kx_fixed, dim=1, keepdim=True)
        
        u_final = sum_cos + sum_sin
        
        return u_final

    def log_specific_params(self, mlflow):
        # Override the base method to log the final learned frequencies.
        print(f"Logging specific parameters for {self.__class__.__name__}")
        if mlflow.active_run():
            final_k_learnable = self.k_fixed.detach().cpu().numpy().flatten()
            
            # Log as individual metrics
            # for i, k_val in enumerate(final_k_learnable):
            #     mlflow.log_metric(f"final_k_learnable_{i}", k_val)
            
            # Log as CSV artifact
            df_k = pd.DataFrame({
            'frequency_index': range(len(final_k_learnable)),
            'k_value': final_k_learnable
            })
            k_path = "final_learned_frequencies.csv"
            df_k.to_csv(k_path, index=False)
            mlflow.log_artifact(k_path, "model_parameters")
            
            print(f"Logged final learned frequencies: {np.sort(final_k_learnable)}")
            print(f"Logged frequencies to artifact: model_parameters/{k_path}")

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
            k_n_final = self.k_fixed
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
            coeffs_path = "full_coefficient_map.csv"
            df_coeffs.to_csv(coeffs_path, index=False)
            mlflow.log_artifact(coeffs_path, "coefficient_data")
            print(f"Logged full coefficient map to artifact: coefficient_data/{coeffs_path}")

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