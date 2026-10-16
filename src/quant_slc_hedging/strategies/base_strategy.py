from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class StrategyAction:
    additional_repayment: np.ndarray
    investment_contribution: np.ndarray


class Strategy:
    """Base strategy format."""

    def decide(self, salary: np.ndarray, loan_balance: np.ndarray) -> StrategyAction:
        return StrategyAction(
            additional_repayment=np.zeros_like(salary),
            investment_contribution=np.zeros_like(salary),
        )

    def investment_growth(self, salary: np.ndarray, observation: int) -> np.ndarray:
        return np.ones_like(salary)

    def loan_payoff_choice(
        self, loan_balance: np.ndarray, investment_balance: np.ndarray
    ) -> np.ndarry:
        return np.zeros_like(loan_balance)
