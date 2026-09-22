from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, SalaryGrowthType, salary_growth_amounts
from quant_slc_hedging.salary import SalaryModel
from quant_slc_hedging.loan import LoanModel
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from quant_slc_hedging.strategies.base_strategy import Strategy, StrategyAction

class MinRepaymentStrategy(Strategy):
    """Minimum repayment strategy where no additional repayments are made."""

    def decide(self, salary: np.ndarray, loan_balance: np.ndarray) -> StrategyAction:
        return StrategyAction(
            additional_repayment=np.zeros_like(salary),
            investment_contribution=np.zeros_like(salary)
        )