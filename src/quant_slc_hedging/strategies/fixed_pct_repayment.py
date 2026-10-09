from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, SalaryGrowthType, salary_growth_amounts, InvestmentModelInputs
from quant_slc_hedging.salary import SalaryModel
from quant_slc_hedging.loan import LoanModel
from quant_slc_hedging.strategies.base_strategy import Strategy, StrategyAction
import numpy as np 
import pandas as pd
from dataclasses import dataclass

class FixedPctRepaymentStrategy(Strategy):
    """Repayment strategy that pays a fixed % of salary each month if above threshold."""
    def __init__(self, fixed_excess_pct: float, repayment_threshold: float, investment_config: Optional[InvestmentModelInputs] = None) -> None:
        self.fixed_excess_pct = fixed_excess_pct
        self.repayment_threshold = repayment_threshold
        self.investment_config = investment_config
        self.annualisation_factor = 12.0

    def decide(self, salary: np.ndarray, loan_balance: np.ndarray) -> StrategyAction:
        """Returns a monthly additional repayment from an input annual salary."""
        salary_excess = np.maximum(0, salary - self.repayment_threshold)
        additional_repayment = salary_excess * self.fixed_excess_pct / 12.0

        return StrategyAction(
            additional_repayment=additional_repayment,
            investment_contribution=np.zeros_like(salary)
        )
    
    def investment_growth(self, salary: np.ndarray, observation: int) -> np.ndarray:
        if self.investment_config:
            return np.ones_like(salary) + (self.investment_config.annual_risk_free_rate / self.annualisation_factor)
        else:
            return np.ones_like(salary)
