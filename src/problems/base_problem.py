from abc import ABC, abstractmethod

class BaseProblem(ABC):
    def __init__(self, config):
        self.config = config
        self.geomtime = self.setup_domain()

    @property
    @abstractmethod
    def x_min(self):
        pass

    @property
    @abstractmethod
    def x_max(self):
        pass

    @abstractmethod
    def setup_domain(self):
        pass

    @abstractmethod
    def pde(self, inputs, outputs):
        pass

    @abstractmethod
    def get_ics_bcs(self):
        pass

    @abstractmethod
    def analytical_solution(self, xt):
        pass

    @abstractmethod
    def get_initial_condition_func(self):
        pass
        
    @abstractmethod
    def get_plot_amplitude(self):
        pass

    @property
    def domain_length(self):
        return self.x_max - self.x_min