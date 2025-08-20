import torch
import numpy as np
import pandas as pd
import mlflow
import deepxde as dde
import os
import plotly.express as px

class MLFlowMetricsLogger(dde.callbacks.Callback):
    """
    A generic DDE callback to log training/testing losses and metrics to MLflow.
    This callback is problem-agnostic and model-agnostic.
    """
    def __init__(self, log_every=100):
        super().__init__()
        self.log_every = log_every
        self._last_log_step = -1

    def on_train_begin(self):
        """Log the initial state at step 0."""
        self._last_log_step = -1
        # It's better to log after the first step, so we'll do it in on_epoch_end
        
    def on_epoch_end(self):
        """Logs metrics at the end of an epoch (which in DDE is one training step)."""
        current_step = self.model.train_state.step
        
        # Throttle logging: only log if it's the first step (step 0), or if
        # enough steps have passed since the last log.
        if current_step == 0 or (current_step - self._last_log_step >= self.log_every):
             self._log_current_metrics()

    def on_train_end(self):
        """Ensures the metrics from the final training step are always logged."""
        print("Logging final metrics at the end of training.")
        self._log_current_metrics()

    def _log_current_metrics(self):
        """Helper function to perform the actual logging to avoid code duplication."""
        if not (self.model.losshistory and self.model.losshistory.steps):
            return

        current_step = self.model.train_state.step

        # Avoid re-logging the same step if called from multiple hooks
        if current_step == self._last_log_step and current_step != 0:
            return

        hist = self.model.losshistory
        idx = -1  # Use the last recorded entry
        log_data = {}

        # --- Dynamically name and log loss components ---
        comp_names = []
        num_losses = len(hist.loss_train[idx])
        if hasattr(self.model.data, 'pde') and self.model.data.pde is not None:
            comp_names.append('PDE_loss')
        
        num_icbc_losses = num_losses - len(comp_names)
        if num_icbc_losses > 0:
            if hasattr(self.model.data, 'bcs') and len(self.model.data.bcs) == num_icbc_losses:
                bc_count, ic_count, op_count = 0, 0, 0
                for icbc in self.model.data.bcs:
                    if isinstance(icbc, dde.icbc.OperatorBC): op_count += 1; comp_names.append(f'IC_u_t_loss_{op_count}')
                    elif isinstance(icbc, dde.icbc.BC): bc_count += 1; comp_names.append(f'BC_loss_{bc_count}')
                    elif isinstance(icbc, dde.icbc.IC): ic_count += 1; comp_names.append(f'IC_loss_{ic_count}')
                    else: comp_names.append('Aux_loss')
            else:
                comp_names.extend([f'aux_loss_{i+1}' for i in range(num_icbc_losses)])

        # Log training losses
        loss_train_current = [loss for loss in hist.loss_train[idx]]
        if len(loss_train_current) == len(comp_names):
            for name, loss_val in zip(comp_names, loss_train_current):
                log_data[f'train_{name}'] = loss_val
        if len(loss_train_current) > 0:
            log_data['loss_train_total'] = sum(loss_train_current)

        # Log testing losses
        if hist.loss_test and len(hist.loss_test) > idx and hist.loss_test[idx] is not None:
            loss_test_current = [loss for loss in hist.loss_test[idx]]
            if len(loss_test_current) == len(comp_names):
                 for name, loss_val in zip(comp_names, loss_test_current):
                    log_data[f'test_{name}'] = loss_val
            if len(loss_test_current) > 0:
                 log_data['loss_test_total'] = sum(loss_test_current)
        
        # Log metrics (like L2 relative error)
        if hist.metrics_test and len(hist.metrics_test) > idx and hist.metrics_test[idx] is not None and self.model.metrics:
             metrics_test_current = hist.metrics_test[idx]
             for metric_func, metric_val in zip(self.model.metrics, metrics_test_current):
                 metric_name = metric_func.__name__ if hasattr(metric_func, '__name__') else "metric"
                 log_data[f'metric_test_{metric_name}'] = metric_val

        if log_data and mlflow.active_run():
            try:
                mlflow.log_metrics(log_data, step=current_step)
                self._last_log_step = current_step
            except Exception as e:
                print(f"Warning: MLflow logging failed at step {current_step}. Error: {e}")


class PredictionLogger(dde.callbacks.Callback):
    """
    Logs the model's predictions on a fixed grid of points throughout training.
    At the end of training, it saves the entire history as a CSV artifact.
    """
    def __init__(self, log_points, log_every=500, run_name=""):
        """
        Args:
            log_points (np.ndarray): A fixed (N, D) array of points to evaluate predictions on.
            log_every (int): Log predictions every N steps.
            run_name (str): A name for the run to identify the artifact.
        """
        super().__init__()
        self.log_points = log_points
        self.log_every = log_every
        self.run_name = run_name
        self.history = None

    def on_train_begin(self):
        if self.history is None:
            self.history = []
        current_step = self.model.train_state.step
        self._log_predictions(current_step)

    def on_epoch_end(self):
        current_step = self.model.train_state.step
        if current_step > 0 and current_step % self.log_every == 0:
            self._log_predictions(current_step)

    def on_train_end(self):
        # Ensure the final step is logged
        final_step = self.model.train_state.step
        if not self.history or self.history[-1]['step'] != final_step:
            self._log_predictions(final_step)
            
        print("Saving prediction history to CSV artifact...")
        if not self.history or mlflow.active_run() is None:
            return

        # Convert history to a structured DataFrame
        all_data = []
        for entry in self.history:
            step = entry['step']
            preds = entry['predictions']
            df_step = pd.DataFrame({
                'time': np.round(self.log_points[:, 1], 4),
                'x': self.log_points[:, 0],
                'model': preds[:, 0],
                'step': step
            })
            all_data.append(df_step)
        
        full_df = pd.concat(all_data, ignore_index=True)
            
        csv_filename = f"prediction_history.csv"
        full_df.to_csv(csv_filename, index=False)
        mlflow.log_artifact(csv_filename, "data_artifacts")
        print(f"Logged prediction history to {csv_filename}")

    def _log_predictions(self, step):
        try:
            predictions_np = self.model.predict(self.log_points)
            self.history.append({'step': step, 'predictions': predictions_np.copy()})
        except Exception as e:
            print(f"Warning: Failed to log predictions at step {step}. Error: {e}")


class ModelParameterLogger(dde.callbacks.Callback):
    """
    Logs the evolution of specific, learnable model parameters (e.g., k_n, w_n).
    This callback is model-aware and relies on the model having specific methods.
    """
    def __init__(self, net, log_every=500, run_name=""):
        super().__init__()
        self.net = net # Direct reference to the network module
        self.log_every = log_every
        self.run_name = run_name
        self.history = None

    def on_train_begin(self):
        if self.history is None:
            self.history = {}
        current_step = self.model.train_state.step
        self._log_params(current_step)

    def on_epoch_end(self):
        current_step = self.model.train_state.step
        if current_step > 0 and current_step % self.log_every == 0:
            self._log_params(current_step)

    def on_train_end(self):
        # Ensure final parameters are logged
        final_step = self.model.train_state.step
        self._log_params(final_step)

        print("Saving parameter history and creating evolution plots...")
        if not self.history or mlflow.active_run() is None:
            return
        
        # Process and log each type of parameter found in history
        for param_name, data_list in self.history.items():
            if param_name == 'steps':
                continue
            
            steps = self.history['steps']
            # Create DataFrame and save as CSV artifact
            df = pd.DataFrame(data_list, index=steps)
            df.columns = [f'{param_name}_{i}' for i in range(df.shape[1])]
            df.index.name = 'step'
            
            csv_filename = f"{param_name}_history.csv"
            df.to_csv(csv_filename)
            mlflow.log_artifact(csv_filename, "data_artifacts")
            
            # Create evolution plot
            df.reset_index(inplace=True)
            df_melt = df.melt(id_vars='step', var_name='param_label', value_name='value')
            fig = px.line(df_melt, x='step', y='value', color='param_label', 
                          title=f'Evolution of Learned Parameter: {param_name}')
            mlflow.log_figure(fig, f"param_evolution/{param_name}_evolution.html")

    def _log_params(self, step):
        # Check if steps list exists, if not, create it
        if 'steps' not in self.history:
            self.history['steps'] = []
        
        # Avoid duplicate logging for the same step
        if self.history['steps'] and self.history['steps'][-1] == step:
            return
            
        self.history['steps'].append(step)

        with torch.no_grad():
            # Check for k_n (spatial frequencies)
            if hasattr(self.net, 'generate_k_n'):
                if 'k_n' not in self.history: self.history['k_n'] = []
                current_k_n = self.net.generate_k_n().cpu().numpy().squeeze()
                self.history['k_n'].append(current_k_n.copy())

            # Check for w_n (temporal frequencies)
            if hasattr(self.net, 'generate_w_n'):
                if 'w_n' not in self.history: self.history['w_n'] = []
                current_w_n = self.net.generate_w_n().cpu().numpy().squeeze()
                self.history['w_n'].append(current_w_n.copy())