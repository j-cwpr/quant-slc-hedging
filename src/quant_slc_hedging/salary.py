from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, SalaryGrowthType, salary_growth_amounts
import numpy as np 
import pandas as pd
from typing import Tuple

# Model salary as S(t+1)=S(t)e^(mu + sigma * rand)

class SalaryModel:
    def __init__(self, config: SalaryModelInputs, rng_gen: np.random.Generator) -> None:
        self.config = config
        self.rng_gen = rng_gen
        self._build_params()

    def _build_params(self) -> None:
        if self.config.salary_growth_dist.growth_type == "Custom":
            annual_rate, annual_vol = self.config.salary_growth_dist.custom
        else:
            annual_rate, annual_vol = salary_growth_amounts[self.config.salary_growth_dist.growth_type]
        monthly_rate = (1 + annual_rate)**(1/12) - 1
        monthly_vol = annual_vol / 12**0.5
        self.sigma = monthly_vol
        self.mu = np.log(1 + monthly_rate) - 0.5 * self.sigma**2

    def _generate_random_array(self, size: Tuple[int, int]) -> np.ndarray:
        return self.rng_gen.standard_normal(size)

    def _build_paths(self, random_grid: np.ndarray) -> np.ndarray:
        n_paths, n_obs = random_grid.shape 

        salary_paths = np.empty(
            (n_paths, n_obs + 1),
            dtype=float
        )
        salary_paths[:, 0] = self.config.starting_salary

        growth = np.exp(self.mu + self.sigma * random_grid)

        salary_paths[:, 1:] = salary_paths[:, [0]] * np.cumprod(growth, axis=1)

        return salary_paths


    def generate_salary_paths(self, n_paths: int, n_months: int) -> np.ndarray:
        rand_grid = self._generate_random_array((n_paths, n_months - 1))
        salary_paths = self._build_paths(rand_grid)

        return salary_paths

# if __name__ == '__main__':
#     config = SalaryModelInputs(
#     starting_salary=50_000,
#     starting_loan_balance=60_000,
#     salary_growth_dist=SalaryGrowthType(growth_type='High')
#     )
#     seed = 1234
#     rng_gen = np.random.default_rng(seed)
#     sm = SalaryModel(config, rng_gen)
#     sp = sm.generate_salary_paths(1000, 30*12)