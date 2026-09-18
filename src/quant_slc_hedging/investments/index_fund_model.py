from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, InvestmentModelInputs, salary_growth_amounts
from quant_slc_hedging.salary import SalaryModel
from quant_slc_hedging.loan import LoanModel
from quant_slc_hedging.strategies.base_strategy import Strategy, StrategyAction
import numpy as np 
import pandas as pd
from dataclasses import dataclass

# For a given fund we may expect an annual return and vol
# For many paths find the monthly return on investment or
# the growth of investment

class IndexFundModel:
    def __init__(self, config: InvestmentModelInputs, rng_gen: np.random.Generator) -> None:
        self.config = config
        self.rng_gen = rng_gen
        self._build_params()

    def _build_params(self) -> None:
        annual_rate, annual_vol = self.config.annual_expected_return, self.config.annual_vol
        monthly_rate = (1 + annual_rate)**(1/12) - 1
        monthly_vol = annual_vol / 12**0.5
        self.sigma = monthly_vol
        self.mu = np.log(1 + monthly_rate) - 0.5 * self.sigma**2

    def _generate_random_array(self, size: Tuple[int, int]) -> np.ndarray:
        return self.rng_gen.standard_normal(size)

    def generate_growth_paths(self, n_paths: int, n_months: int) -> np.ndarray:
        rand_grid = self._generate_random_array((n_paths, n_months - 1))
        growth_paths = np.exp(self.mu + self.sigma * rand_grid)

        return growth_paths

# if __name__ == '__main__':
#     config = InvestmentModelInputs(
#         initial_investment_balance=50_000,
#         annual_expected_return=0.07,
#         annual_vol=0.11
#     )
#     seed = 1234
#     rng_gen = np.random.default_rng(seed)
#     fm = IndexFundModel(config, rng_gen)
#     fgp = fm.generate_growth_paths(1000, 30*12)
#     print("done")