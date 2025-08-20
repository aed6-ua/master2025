import deepxde as dde
import torch
import numpy as np

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
        self.optimizer_name = self.config.get('optimizer', 'Adam')
        self.lr = self.config.get('learning_rate', 0.001)
        self.loss_weights = self.config.get('loss_weights', None) # Can be None
        self.iterations = self.config.get('iterations', 10000)
        self.use_lbfgs = self.config.get('use_lbfgs', False)
        self.lbfgs_maxiter = self.config.get('lbfgs_maxiter', 10000)

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
            
        return losshistory, train_state