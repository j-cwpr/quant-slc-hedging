from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, SalaryGrowthType, salary_growth_amounts, InvestmentModelInputs
from quant_slc_hedging.salary import SalaryModel
from quant_slc_hedging.loan import LoanModel
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from quant_slc_hedging.strategies.base_strategy import Strategy, StrategyAction
from typing import Optional

class MinRepaymentStrategy(Strategy):
    """Minimum repayment strategy where no additional repayments are made."""

    def __init__(self, investment_config: Optional[InvestmentModelInputs] = None) -> None:
        self.investment_config = investment_config
        self.annualisation_factor = 12

    def decide(self, salary: np.ndarray, loan_balance: np.ndarray) -> StrategyAction:
        return StrategyAction(
            additional_repayment=np.zeros_like(salary),
            investment_contribution=np.zeros_like(salary)
        )
    def investment_growth(self, salary: np.ndarray, observation: int) -> np.ndarray:
        if self.investment_config:
            return np.ones_like(salary) + (self.investment_config.annual_risk_free_rate / self.annualisation_factor)
        else:
            return np.ones_like(salary)
