from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, InvestmentModelInputs, salary_growth_amounts
from quant_slc_hedging.salary import SalaryModel
from quant_slc_hedging.loan import LoanModel
from quant_slc_hedging.strategies.base_strategy import Strategy, StrategyAction
from quant_slc_hedging.investments.index_fund_model import IndexFundModel
import numpy as np 
import pandas as pd
from dataclasses import dataclass

class IndexFundStrategy(Strategy):
    def __init__(self, investment_config: InvestmentModelInputs, investment_pct: float, repayment_threshold: float, rng_gen: np.random.Generator, n_paths: int, n_obs: int) -> None:
        self.investment_pct = investment_pct
        self.repayment_threshold = repayment_threshold
        self.config = investment_config
        self.index_model = IndexFundModel(
            config=investment_config,
            rng_gen=rng_gen
        )
        self.growth_rates = self.index_model.generate_growth_paths(n_paths=n_paths, n_months=n_obs)

    def decide(self, salary: np.ndarray, loan_balance: np.ndarray) -> StrategyAction:
        excess_salary = np.maximum(salary - self.repayment_threshold, 0)
        investment_contribution = excess_salary * self.investment_pct / 12

        return StrategyAction(
            additional_repayment=np.zeros_like(salary),
            investment_contribution=investment_contribution
        )

    def investment_growth(self, salary: np.ndarray, observation: int) -> np.ndarray:
        return self.growth_rates[:, observation-1]

    def loan_payoff_choice(self, loan_balance: np.ndarray, investment_balance: np.ndarray) -> np.ndarray:
        if self.config.payoff_loan_with_investments:
            payoff = np.where(investment_balance >= loan_balance, loan_balance, 0)
        
        return payoff

