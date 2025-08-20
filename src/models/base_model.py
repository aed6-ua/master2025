import deepxde as dde

class BaseModel(dde.nn.NN):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_parameters = 0

    def log_specific_params(self, _):
        """
        Optional method for models to log their specific learned parameters (like k_n).
        """
        print(f"No specific parameter logging implemented for {self.__class__.__name__}")
        pass