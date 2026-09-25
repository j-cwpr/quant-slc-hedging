from quant_slc_hedging.data_model import LoanModelInputs, SalaryModelInputs, InvestmentModelInputs, SalaryGrowthType, salary_growth_amounts
from quant_slc_hedging.salary import SalaryModel
from quant_slc_hedging.loan import LoanModel, LoanModelResult
from quant_slc_hedging.strategies.base_strategy import Strategy
from quant_slc_hedging.strategies.min_repayment import MinRepaymentStrategy
from quant_slc_hedging.strategies.fixed_pct_repayment import FixedPctRepaymentStrategy
from quant_slc_hedging.strategies.index_fund import IndexFundStrategy
from quant_slc_hedging.strategies.covered_call import CoveredCallStrategy
import numpy as np 
import pandas as pd
from typing import List, Union
from dataclasses import dataclass

# One SimHandler run performs n_paths sims for a salary config an strategy

@dataclass
class SimulationResult:
    salary_paths: np.ndarray
    loan_result: LoanModelResult
    investment_contribution: np.ndarray
    investment_balance: np.ndarray

class SimulationHandler:
    def __init__(self, salary_config: SalaryModelInputs, loan_config: LoanModelInputs, investment_config: InvestmentModelInputs, strategy: Strategy, n_paths: int, observations: int, salary_rng: np.random.Generator) -> None:
        self.salary_config = salary_config
        self.loan_config = loan_config
        self.investment_config = investment_config
        self.strategy = strategy
        self.n_paths = n_paths
        self.observations = observations
        self.salary_model = SalaryModel(salary_config, salary_rng)
        self.loan_model = LoanModel(loan_config)

    def run_simulation(self):
        salary_paths = self.salary_model.generate_salary_paths(n_paths=self.n_paths, n_months=self.observations)
        loan_balance = np.zeros((self.n_paths, self.observations), dtype=float)
        interest_accrued = np.zeros((self.n_paths, self.observations), dtype=float)
        base_repayment = np.zeros((self.n_paths, self.observations), dtype=float)
        loan_balance[:, 0] = self.loan_config.initial_loan_balance
        additional_repayments = np.zeros((self.n_paths, self.observations), dtype=float)
        investment_contributions = np.zeros((self.n_paths, self.observations), dtype=float)
        investment_balances = np.zeros((self.n_paths, self.observations), dtype=float)
        investment_balances[:, 0] = self.investment_config.initial_investment_balance

        for obs in range(1, self.observations):
            salary = salary_paths[:, obs]
            prev_loan_balance = loan_balance[:, obs-1]

            # Strategy action for mandatory/additional repayment
            action = self.strategy.decide(
                salary=salary,
                loan_balance=prev_loan_balance
            )

            # Calculate investments balance
            investment_growth = self.strategy.investment_growth(salary=salary, observation=obs)
            investment_balances[:, obs] = (investment_balances[:, obs - 1] + action.investment_contribution) * investment_growth

            # Calculate loan
            loan_result = self.loan_model.calculate_obs(prev_loan_balance=prev_loan_balance, salaries=salary, additional_repayment=action.additional_repayment)

            # If investment balance enough then payoff the loan
            payoff = self.strategy.loan_payoff_choice(loan_balance=loan_result.loan_balance, investment_balance=investment_balances[:, obs])

            # Update arrays for next obs
            loan_balance[:, obs] = loan_result.loan_balance - payoff
            investment_balances[:, obs] -= payoff
            interest_accrued[:, obs] = loan_result.interest_accrued
            base_repayment[:, obs] = loan_result.base_repayment
            additional_repayments[:, obs] = loan_result.additional_repayment + payoff
            investment_contributions[:, obs] = action.investment_contribution
            
        
        print("Sim finished")

        return SimulationResult(
            salary_paths=salary_paths,
            loan_result=LoanModelResult(
                loan_balance=loan_balance,
                interest_accrued=interest_accrued,
                base_repayment=base_repayment,
                additional_repayment=additional_repayments
            ),
            investment_contribution=investment_contributions,
            investment_balance=investment_balances
        )

        # pd.DataFrame(loan_balance[:, -1]).describe()



if __name__ == "__main__":
    n_paths = 10_000
    years_remaining = 30
    observations = years_remaining * 12
    seed = 1234
    starting_salary = 35_000
    initial_loan = 45_000
    initial_investment_balance = 10_000
    # salary_growth = SalaryGrowthType(growth_type="Medium")
    salary_growth = SalaryGrowthType(growth_type="Custom", custom=(0.5, 0.3))
    salary_config=SalaryModelInputs(
        starting_salary=starting_salary,
        salary_growth_dist=salary_growth
        )
    loan_config = LoanModelInputs(
        initial_loan_balance=initial_loan,
        remaining_loan_term_months=12*years_remaining
    )
    investment_config = InvestmentModelInputs(
        initial_investment_balance=initial_investment_balance,
        annual_expected_return=0.07,
        annual_vol=0.20,
        annual_risk_free_rate=0.05,
        payoff_loan_with_investments=True
    )
    salary_rng = np.random.default_rng(seed)
    investment_rng = np.random.default_rng(seed + 1) 
    # strategy = MinRepaymentStrategy()
    # strategy = FixedPctRepaymentStrategy(fixed_excess_pct=0.05, repayment_threshold=loan_config.repayment_threshold)
    # strategy = IndexFundStrategy(investment_config=investment_config, investment_pct=0.05, repayment_pct=0.05, repayment_threshold=loan_config.repayment_threshold, rng_gen=investment_rng, n_paths=n_paths, n_obs=observations)
    strategy = CoveredCallStrategy(investment_config=investment_config, investment_pct=0.05, repayment_pct=0.05, repayment_threshold=loan_config.repayment_threshold, target_moneyness=1.1, rng_gen=investment_rng, n_paths=n_paths, n_obs=observations)
    sim = SimulationHandler(salary_config=salary_config, loan_config=loan_config, investment_config=investment_config, strategy=strategy, n_paths=n_paths, observations=observations, salary_rng=salary_rng)
    res = sim.run_simulation()
    print("done")