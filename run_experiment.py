import argparse
import yaml
import mlflow
import pandas as pd
import torch
import deepxde as dde
from importlib import import_module
import random
import numpy as np
from src.utils.analysis import fft1D

torch.set_default_dtype(torch.float64)
dde.config.real.set_float64()

def setup_device(config):
    """Sets up the device for torch and deepxde."""
    device_str = config.get('execution', {}).get('device', 'gpu')
    if device_str.lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("--- Running on CPU ---")
    else:
        device = torch.device("cuda:0")
        print(f"--- Running on GPU: {torch.cuda.get_device_name(0)} ---")
    return device

# --- Helper to dynamically load classes ---
def load_class(module_path, class_name):
    module = import_module(module_path)
    return getattr(module, class_name)

def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # The following two lines are often recommended for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudunn.benchmark = False
    print(f"--- Set all random seeds to {seed} ---")

def main(config_path, seed):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # if seed is not None:
    #     set_random_seed(seed)

    device = setup_device(config)

    # 2. Load Problem, Model, and Trainer classes dynamically
    ProblemClass = load_class('src.problems', config['problem']['name'])
    ModelClass = load_class('src.models.all_models', config['model']['name'])
    TrainerClass = load_class('src.trainers.standard_trainer', config['trainer']['name'])
    
    # 3. Instantiate components
    problem = ProblemClass(config)
    if config['model']['output_transform']:
        net = ModelClass(config, output_transform=problem.get_output_transform()).to(device)
    else:
        net = ModelClass(config).to(device)
    trainer = TrainerClass(config)


    # --- NEW: Pre-training step for models that need IC data ---
    if hasattr(net, 'load_ic_data'):
        print("Model has 'load_ic_data' method. Performing pre-training IC analysis.")
        # Generate a fine grid of points for the IC
        ic_grid_x = np.linspace(0, problem.L, 2048).reshape(-1, 1)
        ic_func = problem.get_initial_condition_func()
        ic_grid_u = ic_func(ic_grid_x).flatten()
        
        # Get number of fixed frequencies from config
        model_cfg = config.get('model', {})
        n_freq_fixed = model_cfg.get('n_freq_k_fixed', 50)
        
        # Perform FFT
        k_vals, a_vals, b_vals = fft1D(ic_grid_u, ic_grid_x.flatten(), n_freq_fixed)
        
        # Load data into the network
        net.load_ic_data(k_vals, a_vals, b_vals)

    # 4. Setup DeepXDE Data object
    data = dde.data.TimePDE(
        problem.geomtime,
        problem.pde,
        problem.get_ics_bcs(),
        num_domain=config['data']['num_domain'],
        num_boundary=config['data']['num_boundary'],
        num_initial=config['data']['num_initial'],
        solution=problem.analytical_solution,
        num_test=config['data']['num_test'],
    )
    
    dde_model = dde.Model(data, net)
    
    # 5. Setup MLFlow
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    experiment = mlflow.get_experiment_by_name(config['experiment_name'])
    if not experiment:
        mlflow.create_experiment(name=config['experiment_name'])
        experiment = mlflow.get_experiment_by_name(config['experiment_name'])

    base_run_name = config['run_name']
    run_name_with_seed = f"{base_run_name}_seed_{seed}" if seed is not None else base_run_name

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name_with_seed) as run:
        mlflow.log_params(config)
        if seed is not None:
            mlflow.log_param("random_seed", seed)
        mlflow.log_param("num_model_parameters", sum(p.numel() for p in net.parameters()))
        
        # Setup Callbacks
        from src.utils.callbacks import MLFlowMetricsLogger, PredictionLogger, ModelParameterLogger

        log_points_np = problem.geomtime.uniform_points(100*100)
    
        # This callback logs losses and L2 error
        metrics_logger = MLFlowMetricsLogger(log_every=300)
        
        # This callback logs prediction history for creating animations later
        prediction_logger = PredictionLogger(
            log_points=log_points_np, 
            log_every=300, 
            run_name=config['run_name']
        )
        
        # This callback logs learned parameters like k_n and w_n
        param_logger = ModelParameterLogger(
            net=net,
            log_every=300,
            run_name=config['run_name']
        )

        # Resampler
        pde_resampler = dde.callbacks.PDEPointResampler(period=300)
        
        # --- Combine all callbacks into a list ---
        callbacks = [pde_resampler, metrics_logger, prediction_logger, param_logger]
        
        # 6. Train the model
        if device.type == 'cuda':
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

        trainer.train(dde_model, callbacks=callbacks)

        if device.type == 'cuda':
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0
            mlflow.log_metric("training_time_seconds", elapsed_time)
        
        # 7. Final Logging and Visualization
        print("--- Final Logging and Visualization ---")
        
        # Log learned parameters specific to the model
        net.log_specific_params(mlflow)
        
        # Ensure log_points are on the correct device for prediction
        log_points_tensor = torch.from_numpy(log_points_np).to(device)
        final_preds_tensor = torch.from_numpy(dde_model.predict(log_points_np)).to(device)
        true_vals_tensor = problem.analytical_solution(log_points_tensor)

        # Move results back to CPU for pandas/plotting
        final_preds = final_preds_tensor.cpu().numpy()
        true_vals = true_vals_tensor.cpu().numpy()
        df = pd.DataFrame({
            'x': log_points_np[:, 0],
            'time': log_points_np[:, 1],
            'model': final_preds.flatten(),
            'ground_truth': true_vals.flatten(),
            'difference': (final_preds - true_vals).flatten()
        })
        from src.utils.plotting import create_final_solution_plots
        create_final_solution_plots(df, config['run_name'], problem.plot_amplitude)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config YAML file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the experiment.")
    args = parser.parse_args()
    main(args.config, args.seed)