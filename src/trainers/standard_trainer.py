import deepxde as dde
import torch
import numpy as np
import mlflow.pytorch
import pandas as pd

def custom_l2_relative_error(y_true, y_pred):
    """L2 relative error that safely handles GPU tensors."""
    if y_true is None:
        print("ERROR in custom_l2_relative_error: y_true is None!")
        return np.inf # Or some other indicator of error
    if y_pred is None:
        print("ERROR in custom_l2_relative_error: y_pred is None!")
        return np.inf
    if isinstance(y_true, torch.Tensor) and y_true.device.type == 'cuda':
        y_true = y_true.cpu()
    if isinstance(y_pred, torch.Tensor) and y_pred.device.type == 'cuda':
        y_pred = y_pred.cpu()

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().numpy()

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    norm_true = np.linalg.norm(y_true_np)
    if norm_true < np.finfo(y_true_np.dtype).eps:
        return 0.0 if np.linalg.norm(y_pred_np) < np.finfo(y_pred_np.dtype).eps else np.inf
    else:
        return np.linalg.norm(y_true_np - y_pred_np) / norm_true

class StandardTrainer:
    def __init__(self, config):
        self.config = config.get('trainer', {}) # Safely get the trainer config
        self.model_name = config.get('run_name', 'no_name')
        self.optimizer_name = self.config.get('optimizer', 'Adam')
        self.lr = self.config.get('learning_rate', 0.001)
        self.loss_weights = self.config.get('loss_weights', None) # Can be None
        self.iterations = self.config.get('iterations', 10000)
        self.use_lbfgs = self.config.get('use_lbfgs', False)
        self.lbfgs_maxiter = self.config.get('lbfgs_maxiter', 10000)
        self.save_model = self.config.get('save_model', False)

    def train(self, dde_model, callbacks):
        optimizer = torch.optim.Adam(dde_model.net.parameters(), lr=self.lr)
        dde_model.compile(optimizer, loss_weights=self.loss_weights, metrics=[custom_l2_relative_error])
        
        losshistory, train_state = dde_model.train(
            iterations=self.iterations,
            callbacks=callbacks,
            display_every=300
        )

        if self.use_lbfgs and self.lbfgs_maxiter > 0:
            dde.optimizers.config.set_LBFGS_options(maxiter=self.lbfgs_maxiter)
            dde_model.compile("L-BFGS", metrics=[custom_l2_relative_error])
            losshistory, train_state = dde_model.train(callbacks=callbacks, display_every=100)


        if self.save_model:
            print("Saving the trained model to MLflow with signature...")
            
            # 1. Get the model's device and dtype directly from its parameters
            target_device = next(dde_model.net.parameters()).device
            target_dtype = next(dde_model.net.parameters()).dtype

            # 2. Create the example tensor directly on the target device and with the target dtype
            # A small batch of 5 points with the correct shape (N, 2)
            input_example_tensor = torch.rand(5, 2, device=target_device, dtype=target_dtype)

            # 3. Create a pandas DataFrame for the named signature.
            # Convert the tensor to a CPU numpy array first for pandas compatibility.
            input_example_df = pd.DataFrame(
                input_example_tensor.cpu().numpy(),
                columns=["x", "t"]
            )
            
            # 4. Log the model
            mlflow.pytorch.log_model(
                pytorch_model=dde_model.net,
                artifact_path="model",
                input_example=input_example_df, # Pass the DataFrame
                registered_model_name=self.model_name
            )
            print("Model saved successfully.")
            
        return losshistory, train_state
    
class SpectralTrainer(StandardTrainer):
    def __init__(self, config):
        super().__init__(config)
        
    def train(self, dde_model, callbacks):
        optimizer = torch.optim.Adam([
            {'params': dde_model.net.mlp_coeffs_learnable.parameters(), 'lr': self.lr},
            {'params': dde_model.net.k_learnable_params, 'lr': self.lr * 10}
        ])
        dde_model.compile(optimizer, loss_weights=self.loss_weights, metrics=[custom_l2_relative_error])
        
        losshistory, train_state = dde_model.train(
            iterations=self.iterations,
            callbacks=callbacks,
            display_every=300
        )

        if self.use_lbfgs and self.lbfgs_maxiter > 0:
            dde.optimizers.config.set_LBFGS_options(maxiter=self.lbfgs_maxiter)
            dde_model.compile("L-BFGS", metrics=[custom_l2_relative_error])
            losshistory, train_state = dde_model.train(callbacks=callbacks, display_every=100)


        if self.save_model:
            print("Saving the trained model to MLflow with signature...")
            
            # 1. Get the model's device and dtype directly from its parameters
            target_device = next(dde_model.net.parameters()).device
            target_dtype = next(dde_model.net.parameters()).dtype

            # 2. Create the example tensor directly on the target device and with the target dtype
            # A small batch of 5 points with the correct shape (N, 2)
            input_example_tensor = torch.rand(5, 2, device=target_device, dtype=target_dtype)

            # 3. Create a pandas DataFrame for the named signature.
            # Convert the tensor to a CPU numpy array first for pandas compatibility.
            input_example_df = pd.DataFrame(
                input_example_tensor.cpu().numpy(),
                columns=["x", "t"]
            )
            
            # 4. Log the model
            mlflow.pytorch.log_model(
                pytorch_model=dde_model.net,
                artifact_path="model",
                input_example=input_example_df, # Pass the DataFrame
                registered_model_name=self.model_name
            )
            print("Model saved successfully.")
            
        return losshistory, train_state