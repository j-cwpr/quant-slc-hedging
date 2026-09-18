from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, SalaryGrowthType, salary_growth_amounts
import numpy as np 
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LoanModelResult:
    loan_balance: np.ndarray
    interest_accrued: np.ndarray
    base_repayment: np.ndarray
    additional_repayment: Optional[np.ndarray] = None


class LoanModel:
    def __init__(self, config: LoanModelInputs) -> None:
        self.config = config

    def _calculate_var_interest_rate(self, salary: np.ndarray, monthly: bool = True) -> np.ndarray:
        """Calculate the variable interest rate dependent on upper and lower salary thresholds."""
        lower_salary_t = self.config.repayment_threshold
        upper_salary_t = self.config.interest_rate_spread_threshold

        salary_prop = (salary - lower_salary_t) / (upper_salary_t - lower_salary_t)
        salary_prop = np.clip(salary_prop, 0.0, 1.0)

        interest_rate = self.config.base_interest_rate + self.config.interest_rate_spread * salary_prop
        
        if monthly:
            interest_rate = (interest_rate + 1)**(1/12) - 1

        return interest_rate

    def _calculate_base_repayment(self, salary: np.ndarray, monthly: bool = True) -> np.ndarray:
        salary_above_threshold = np.maximum(
            salary - self.config.repayment_threshold, 0
        )

        base_repayment = salary_above_threshold * self.config.repayment_amount_pct_salary

        if monthly:
            base_repayment /= 12

        return base_repayment

    def generate_loan_paths(self, salary_paths: np.ndarray) -> LoanModelResult:
        n_paths, n_obs = salary_paths.shape
        loan_paths = np.zeros((n_paths, n_obs), dtype=float)
        interest_accrued = np.zeros((n_paths, n_obs), dtype=float)
        base_repayment = np.zeros((n_paths, n_obs), dtype=float)
        loan_paths[:, 0] = self.config.initial_loan_balance

        for obs in range(1, n_obs):
            salaries = salary_paths [:, obs]
            interest_rate_monthly = self._calculate_var_interest_rate(salaries, monthly=True)
            base_repayment[:, obs] = self._calculate_base_repayment(salaries, monthly=True)
            interest_accrued[:, obs] = loan_paths[:, obs - 1] * interest_rate_monthly

            # On each observation the loan value = loan previous * interest - base repayment
            loan_paths[:, obs] = np.maximum(
                0,
                loan_paths[:, obs - 1] + interest_accrued[:, obs] - base_repayment[:, obs]
            )

        return LoanModelResult(
            loan_balance=loan_paths,
            interest_accrued=interest_accrued,
            base_repayment=base_repayment
        )

    def calculate_obs(self, prev_loan_balance: np.ndarray, salaries: np.ndarray, additional_repayment: np.ndarry) -> None:
        """Performs a single observation calculation now, needed to properly handle state."""
        interest_rate_monthly = self._calculate_var_interest_rate(salaries, monthly=True)
        base_repayment = self._calculate_base_repayment(salaries, monthly=True)
        interest_accrued = prev_loan_balance * interest_rate_monthly

        # On each observation the loan value = loan previous * interest - base repayment
        loan_balance_before_add = np.maximum(0, prev_loan_balance + interest_accrued - base_repayment)
        actual_add_repayment = np.minimum(additional_repayment, loan_balance_before_add)
        loan_balance = loan_balance_before_add - actual_add_repayment

        return LoanModelResult(
            loan_balance=loan_balance,
            interest_accrued=interest_accrued,
            base_repayment=base_repayment,
            additional_repayment=actual_add_repayment
        )



# if __name__ == '__main__':
#     config = LoanModelInputs(
#         initial_loan_balance=60_000,
#         remaining_loan_term_months=30 * 12,
#         max_loan_term_months= 30 * 12,
#         base_interest_rate= 0.03,
#         interest_rate_spread=0.03,
#         repayment_amount_pct_salary= 0.09,
#         repayment_threshold= 29_385,
#         interest_rate_spread_threshold=52_885
#     )
#     seed = 1234
#     rng_gen = np.random.default_rng(seed)
#     # sp = sm.generate_salary_paths(1000, 30*12)
#     sp = np.vstack([
#         20_000 * (1 + np.arange(0, 100)/100),
#         15_000 * (1 + np.arange(0, 100)/100)
#     ])
#     sp = np.full((1_000, 360), 29_385)
#     ar = np.full((1_000, 360), 0)
#     lm = LoanModel(config)
#     lp = lm.generate_loan_paths(sp)
    
