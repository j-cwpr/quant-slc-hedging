from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, SalaryGrowthType, salary_growth_amounts
import numpy as np 
import pandas as pd
from typing import Tuple, Literal, Union

# Model salary as S(t+1)=S(t)e^(mu + sigma * rand)

class SalaryModel:
    def __init__(self, config: SalaryModelInputs, rng_gen: np.random.Generator, freq: Literal[Union["monthly", "annual"]] = "annual") -> None:
        self.config = config
        self.rng_gen = rng_gen
        self.freq: str = freq

        assert self.freq in ["monthly", "annual"], "Freq is not valid"

        self._build_params()

    def _build_params(self) -> None:
        if self.config.salary_growth_dist.growth_type == "Custom":
            annual_rate, annual_vol = self.config.salary_growth_dist.custom
        else:
            annual_rate, annual_vol = salary_growth_amounts[self.config.salary_growth_dist.growth_type]
        
        if self.freq == "monthly":
            rate = (1 + annual_rate)**(1/12) - 1
            vol = annual_vol / 12**0.5
        else:
            rate = annual_rate
            vol = annual_vol

        self.sigma = vol
        self.mu = np.log(1 + rate) - 0.5 * self.sigma**2

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

    def _build_monthly_paths(self, n_paths: int, n_months: int) -> np.ndarray:
        rand_grid = self._generate_random_array((n_paths, n_months - 1))
        return self._build_paths(rand_grid)

    def _build_annual_paths(self, n_paths: int, n_months: int) -> np.ndarray:
        n_years = int(np.ceil((n_months - 1 )/ 12))
        rand_grid = self._generate_random_array((n_paths, n_years))
        salary_paths = self._build_paths(rand_grid)

        # Convert to monthly observations
        monthly_paths = np.empty((n_paths, n_months))
        monthly_paths[:, 0] = self.config.starting_salary
        
        for year in range(n_years):
            start = year * 12 + 1
            end = min((year+1)*12 + 1 , n_months)
            monthly_paths[:, start: end] = salary_paths[:, year+1, None]
        
        return monthly_paths

    def generate_salary_paths(self, n_paths: int, n_months: int) -> np.ndarray:
        if self.freq == "monthly":
            return self._build_monthly_paths(n_paths, n_months)
        else:
            return self._build_annual_paths(n_paths, n_months)

# if __name__ == '__main__':
#     config = SalaryModelInputs(
#     starting_salary=50_000,
#     salary_growth_dist=SalaryGrowthType(growth_type='Medium')
#     )
#     seed = 1234
#     rng_gen = np.random.default_rng(seed)
#     sm = SalaryModel(config, rng_gen)
#     sp = sm.generate_salary_paths(1000, 30*12)
#     print("Created paths.")