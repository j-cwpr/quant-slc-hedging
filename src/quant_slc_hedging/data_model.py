import pandas as pd 
from dataclasses import dataclass
from typing import Literal, List

# Annual growth rates and vols
salary_growth_amounts = {
    "Low": (0.01, 0.05),
    "Medium": (0.05, 0.1),
    "High": (0.1, 0.12)
}

@dataclass
class SalaryGrowthType:
    growth_type: Literal['High', 'Medium', 'Low', 'Custom'] = 'Medium'
    custom: Optional[Tuple[float]] = None # Allows overrides of mean, vol salary growth

@dataclass 
class SalaryModelInputs:
    starting_salary: float
    salary_growth_dist: SalaryGrowthType

@dataclass
class LoanModelInputs:
    initial_loan_balance: float
    remaining_loan_term_months: int = 30 * 12
    max_loan_term_months: int = 30 * 12
    base_interest_rate: float = 0.03
    interest_rate_spread: float = 0.03
    repayment_amount_pct_salary: float = 0.09
    repayment_threshold: float = 29_385 
    # Between repayment_threshold and interest_rate_spread_threshold
    # the interest rate on the loan is increased pro rata by interest_rate_spread
    interest_rate_spread_threshold: float = 52_885
    # repayment_strategies: List[float] = []

@dataclass
class InvestmentModelInputs:
    initial_investment_balance: float 
    annual_expected_return: float 
    annual_vol: float
    payoff_loan_with_investments: bool
    annual_risk_free_rate: float = 0.03
    

