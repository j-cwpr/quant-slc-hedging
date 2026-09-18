from dataclasses import dataclass
import numpy as np

@dataclass
class StrategyAction:
    additional_repayment: np.ndarray
    investment_contribution: np.ndarray


class Strategy:
    """Base strategy format."""
    def decide(
        self,
        salary: np.ndarray,
        loan_balance: np.ndarray
    ) -> StrategyAction:
        return StrategyAction(
            additional_repayment=np.zeros_like(salary),
            investment_contribution=np.zeros_like(salary)
        )
    
    def investment_growth(self, salary: np.ndarray) -> np.ndarray:
        return np.ones_like(salary)