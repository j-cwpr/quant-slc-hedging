import numpy as np
import pandas as pd

from quant_slc_hedging.simulation import SimulationResult


def calc_stats(
    result: SimulationResult,
    strategy_name: str,
    growth_regime: str,
    summary_only: bool = False,
) -> pd.DataFrame:
    """Calculate summary statistics for one strategy simulation result.

    Assumes:
        result is SimulationResult
    """

    loan = result.loan_result

    loan_balance = np.asarray(loan.loan_balance)
    investment_balance = np.asarray(result.investment_balance)
    investment_contribution = np.asarray(result.investment_contribution)

    terminal_loan = loan_balance[:, -1]
    terminal_investment = investment_balance[:, -1]

    # Terminal wealth before and after write-off
    wealth_pre_writeoff = terminal_investment - terminal_loan
    wealth_post_writeoff = terminal_investment.copy()

    # Cumulative repayments and investment contributions
    base_repayment = np.asarray(loan.base_repayment)

    if loan.additional_repayment is None:
        additional_repayment = np.zeros_like(base_repayment)
    else:
        additional_repayment = np.asarray(loan.additional_repayment)

    total_base_repayment = np.sum(base_repayment, axis=1)
    total_additional_repayment = np.sum(additional_repayment, axis=1)
    total_repayments = total_base_repayment + total_additional_repayment
    total_invested = np.sum(investment_contribution, axis=1)

    # Loan repaid if balance reaches zero at any point
    repaid_before_writeoff = np.any(loan_balance <= 0, axis=1)

    total_interest = np.sum(np.asarray(loan.interest_accrued), axis=1)

    # Top level stats
    summary_stats = {
        "Growth Regime": growth_regime,
        "Strategy": strategy_name,
        "Median Wealth After Write-Off": np.median(wealth_post_writeoff),
        "10th Percentile Wealth After Write-Off": np.percentile(
            wealth_post_writeoff, 10
        ),
        "Probability of Repayment Before Write-Off": np.mean(repaid_before_writeoff),
        "Median Wealth Before Write-Off": np.median(wealth_pre_writeoff),
    }

    if summary_only:
        return pd.DataFrame([summary_stats])

    # Full statistics panel
    full_stats = {
        **summary_stats,
        "Number of Paths": len(terminal_loan),
        "Mean Wealth Before Write-Off": np.mean(wealth_pre_writeoff),
        "10th Percentile Wealth Before Write-Off": np.percentile(
            wealth_pre_writeoff, 10
        ),
        "90th Percentile Wealth Before Write-Off": np.percentile(
            wealth_pre_writeoff, 90
        ),
        "Mean Wealth After Write-Off": np.mean(wealth_post_writeoff),
        "90th Percentile Wealth After Write-Off": np.percentile(
            wealth_post_writeoff, 90
        ),
        "Standard Deviation Before Write-Off": np.std(wealth_pre_writeoff, ddof=1),
        "Standard Deviation After Write-Off": np.std(wealth_post_writeoff, ddof=1),
        "Median Terminal Loan Balance": np.median(terminal_loan),
        "Median Terminal Investment Balance": np.median(terminal_investment),
        "Median Total Repayments": np.median(total_repayments),
        "Median Base Repayments": np.median(total_base_repayment),
        "Median Additional Repayments": np.median(total_additional_repayment),
        "Median Total Invested": np.median(total_invested),
        "Median Total Interest": np.median(total_interest),
        "Probability Balance Remaining At Write-Off": np.mean(terminal_loan > 0),
    }

    return pd.DataFrame([full_stats])
