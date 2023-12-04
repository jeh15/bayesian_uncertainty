from typing import List, Self, Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ml_collections import config_dict


@dataclass
class Parameter:
    param_name: str
    param_range: str
    lower_bound: float
    upper_bound: float
    dy: List[float]
    var_indx: List[int]


class MorrisSampling():
    def __init__(
        self: Self,
        model_fn: Callable,
        params: List[Parameter],
        xdata: npt.ArrayLike,
        config: config_dict.ConfigDict,
    ):
        self.model_fn = model_fn
        self.params = params
        self.xdata = xdata
        self.config = config
        self.num_params = len(params)
        self.reject = 0

    def _theta_sample(
        self: Self,
        param: config_dict.ConfigDict,
        random_variable: npt.ArrayLike,
    ) -> np.ndarray:
        if param.param_range == 'O':
            theta = 10 ** (
                random_variable *
                (param.upper_bound - param.lower_bound) +
                param.lower_bound
            )
        elif param.param_range == 'U':
            theta = (
                random_variable *
                (param.upper_bound - param.lower_bound) +
                param.lower_bound
            )
        elif param.param_range == 'pr':
            theta = param.upper_bound / (random_variable - 1)
        elif param.param_range == 'inf':
            theta = param.upper_bound * np.tan(
                -np.pi / 2 + np.pi * random_variable
            )
        return theta

    def sample(self: Self) -> tuple[np.ndarray, np.ndarray]:
        for i in range(self.config.num_runs):
            good = False
            while not good:
                random_variables = []
                thetas = []
                for param in self.params:
                    random_variable = np.random.rand()
                    theta = self._theta_sample(param, random_variable)
                    random_variables.append(random_variable)
                    thetas.append(theta)
                # Convert list to numpy array:
                random_variables = np.array(random_variables)
                thetas = np.array(thetas)
                response = self.model_fn(thetas, self.xdata)

                if np.min(response) >= self.config.reject_max or np.max(response) <= self.config.reject_min:
                    good = False
                    self.reject += 1
                else:
                    good = True
                    y = np.sum(response, axis=0) / response.shape[0]

            for j in range(self.config.num_iterations):
                step_status = False
                while not step_status:
                    allowed_var = False
                    while not allowed_var:
                        var_indx = np.random.randint(self.num_params)
                        step_size = (np.random.rand() - 0.5) / self.config.step_scale
                        if random_variables[var_indx] + step_size > 0 and random_variables[var_indx] + step_size < 1:
                            allowed_var = True
                            random_variables[var_indx] += step_size
                        else:
                            allowed_var = False
                            self.reject += 1
                    old_var = thetas[var_indx]
                    thetas[var_indx] = self._theta_sample(self.params[var_indx], random_variables[var_indx])
                    y_old = y
                    response = self.model_fn(thetas, self.xdata)
                    step_status = True
                    y = np.sum(response, axis=0) / response.shape[0]

                if (thetas[var_indx] - old_var) != 0:
                    dy = (y_old - y) / (old_var - thetas[var_indx])
                    self.params[var_indx].dy.append(dy)
                    self.params[var_indx].var_indx.append(var_indx)

        means = []
        variances = []
        for param in self.params:
            dy = np.asarray(param.dy)
            mu = np.sum(
                np.abs(dy) / dy.shape[0]
            )
            variance = (np.sum(dy - mu) ** 2) / (dy.shape[0] - 1)
            means.append(mu)
            variances.append(variance)

        means = np.asarray(means)
        variance = np.asarray(variances)

        return means, variance