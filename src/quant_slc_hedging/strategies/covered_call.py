from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, InvestmentModelInputs, salary_growth_amounts
from quant_slc_hedging.salary import SalaryModel
from quant_slc_hedging.loan import LoanModel
from quant_slc_hedging.strategies.base_strategy import Strategy, StrategyAction
from quant_slc_hedging.investments.index_fund_model import IndexFundModel
from quant_slc_hedging.investments.black_scholes_model import BlackScholesModel, BlackScholesModelInput
import numpy as np 
import pandas as pd
from dataclasses import dataclass

# Sell an OTM covered call at a fixed moneyness to generate monthly income

class CoveredCallStrategy(Strategy):
    def __init__(self, investment_config: InvestmentModelInputs, investment_pct: float, repayment_pct: float, repayment_threshold: float, target_moneyness: float, rng_gen: np.random.Generator, n_paths: int, n_obs: int) -> None:
        self.investment_pct = investment_pct
        self.repayment_pct = repayment_pct
        self.repayment_threshold = repayment_threshold
        self.target_moneyness = target_moneyness
        self.config = investment_config
        self.index_model = IndexFundModel(
            config=investment_config,
            rng_gen=rng_gen
        )
        self.fund_levels = self.index_model.generate_fund_levels(n_paths=n_paths, n_months=n_obs)
        self.option_model = BlackScholesModel(
            config=BlackScholesModelInput(
                risk_free_rate=investment_config.annual_risk_free_rate,
                vol=investment_config.annual_vol
            )
        )

    def decide(self, salary: np.ndarray, loan_balance: np.ndarray) -> StrategyAction:
        excess_salary = np.maximum(salary - self.repayment_threshold, 0)
        investment_contribution = excess_salary * self.investment_pct / 12.0
        additional_repayment = excess_salary * self.repayment_pct / 12.0

        return StrategyAction(
            additional_repayment=additional_repayment,
            investment_contribution=investment_contribution
        )

    def investment_growth(self, salary: np.ndarray, observation: int) -> np.ndarray:
        s_next = self.fund_levels[:, observation]
        s = self.fund_levels[:, observation-1]
        strike = s * self.target_moneyness
        premium = self.option_model.call_price(
            underlying=s, 
            strike=strike,
            time_to_exp=1/12
            )
        payoff = np.maximum(0, s_next - strike)
        covered_call_value = s_next + premium - payoff
        growth_rate = covered_call_value / s
        
        return growth_rate

    def loan_payoff_choice(self, loan_balance: np.ndarray, investment_balance: np.ndarray) -> np.ndarray:
        if self.config.payoff_loan_with_investments:
            payoff = np.where(investment_balance >= loan_balance, loan_balance, 0)
        
        return payoff

