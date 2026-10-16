from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_strategy_comparison(
    results: Dict, regimes: List[str] = ["Low", "Medium", "High"]
):
    """
    Plot median post-write-off wealth by strategy and salary-growth regime,
    with 10th–90th percentile error bars.

    Assumes:
        results[growth_regime][strategy_name] = SimulationResult

    Post-write-off wealth is defined as terminal investment balance,
    ignoring the remaining loan balance.
    """

    summary = []

    for regime in regimes:
        for strategy_name, result in results[regime].items():
            terminal_wealth = np.asarray(result.investment_balance)[:, -1]

            summary.append(
                {
                    "Growth Regime": regime,
                    "Strategy": strategy_name,
                    "Median Wealth": np.median(terminal_wealth),
                    "P10": np.percentile(terminal_wealth, 10),
                    "P90": np.percentile(terminal_wealth, 90),
                }
            )

    stats = pd.DataFrame(summary)

    strategies = list(results[regimes[0]].keys())
    regimes_present = [r for r in regimes if r in results]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(strategies))
    width = 0.25

    for i, regime in enumerate(regimes_present):
        regime_stats = (
            stats[stats["Growth Regime"] == regime]
            .set_index("Strategy")
            .reindex(strategies)
        )

        medians = regime_stats["Median Wealth"].to_numpy()
        lower = medians - regime_stats["P10"].to_numpy()
        upper = regime_stats["P90"].to_numpy() - medians

        positions = x + (i - (len(regimes_present) - 1) / 2) * width

        ax.bar(
            positions,
            medians,
            width=width,
            label=regime,
        )

        ax.errorbar(
            positions,
            medians,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )

    ax.set_title("Terminal Wealth After Write-Off by Strategy and Salary Growth")
    ax.set_ylabel("Median Wealth After Loan Write-Off (£)")
    ax.set_xlabel("Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha="right")
    ax.axhline(0, linewidth=0.8)
    ax.legend(title="Salary Growth")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return stats


def plot_writeoff_impact(
    results: Dict,
    regimes: List[str] = ["Low", "Medium", "High"],
    strategy: str = "Minimum Repayment",
):
    """
    Compare median terminal wealth before and after loan write-off.
    """

    rows = []

    for regime in regimes:
        result = results[regime][strategy]

        terminal_loan = np.asarray(result.loan_result.loan_balance)[:, -1]

        terminal_investment = np.asarray(result.investment_balance)[:, -1]

        wealth_pre = terminal_investment - terminal_loan
        wealth_post = terminal_investment

        rows.append(
            {
                "Growth Regime": regime,
                "Median Wealth Before Write-Off": np.median(wealth_pre),
                "Median Wealth After Write-Off": np.median(wealth_post),
            }
        )

    stats = pd.DataFrame(rows)

    x = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x - width / 2,
        stats["Median Wealth Before Write-Off"],
        width,
        label="Before Write-Off",
    )

    ax.bar(
        x + width / 2,
        stats["Median Wealth After Write-Off"],
        width,
        label="After Write-Off",
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_xlabel("Salary Growth Regime")
    ax.set_ylabel("Median Terminal Wealth (£)")
    ax.set_title(f"Impact of Student Loan Write-Off on Terminal Wealth: {strategy}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return stats


def plot_loan_balance_sensitivity(
    sensitivity_results: Dict,
    growth_regime="Medium",
):
    """
    Plot median post-write-off wealth against initial loan balance
    for each strategy in a selected salary-growth regime.

    Assumes:
        sensitivity_results[regime][loan_balance][strategy] = SimulationResult
    """

    regime_results = sensitivity_results[growth_regime]
    loan_balances = sorted(regime_results.keys())

    rows = []

    for initial_loan in loan_balances:
        for strategy_name, result in regime_results[initial_loan].items():
            terminal_wealth = np.asarray(result.investment_balance)[:, -1]

            rows.append(
                {
                    "Initial Loan Balance": initial_loan,
                    "Strategy": strategy_name,
                    "Median Wealth": np.median(terminal_wealth),
                    "P10 Wealth": np.percentile(terminal_wealth, 10),
                    "P90 Wealth": np.percentile(terminal_wealth, 90),
                }
            )

    stats = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in stats["Strategy"].unique():
        subset = stats[stats["Strategy"] == strategy].sort_values(
            "Initial Loan Balance"
        )

        ax.plot(
            subset["Initial Loan Balance"] / 1000,
            subset["Median Wealth"] / 1000,
            marker="o",
            label=strategy,
        )

    ax.set_xlabel("Initial Loan Balance (£000s)")
    ax.set_ylabel("Median Post-Write-Off Wealth (£000s)")
    ax.set_title(f"Loan Balance Sensitivity — {growth_regime} Salary Growth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

    return stats


def plot_repayment_probability_sensitivity(sensitivity_results: Dict):
    """
    Plot probability of repaying the loan before write-off against
    initial loan balance, with one subplot per salary-growth regime.
    """

    regimes = ["Low", "Medium", "High"]
    rows = []

    for regime in regimes:
        for initial_loan, strategy_results in sensitivity_results[regime].items():
            for strategy_name, result in strategy_results.items():
                loan_balance = np.asarray(result.loan_result.loan_balance)

                # True if the balance reaches zero at any time
                repaid = np.any(loan_balance <= 0, axis=1)

                rows.append(
                    {
                        "Growth Regime": regime,
                        "Initial Loan Balance": initial_loan,
                        "Strategy": strategy_name,
                        "Repayment Probability": np.mean(repaid) * 100,
                    }
                )

    stats = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        sharey=True,
    )

    for ax, regime in zip(axes, regimes):
        regime_stats = stats[stats["Growth Regime"] == regime]

        for strategy in regime_stats["Strategy"].unique():
            subset = regime_stats[regime_stats["Strategy"] == strategy].sort_values(
                "Initial Loan Balance"
            )

            ax.plot(
                subset["Initial Loan Balance"] / 1000,
                subset["Repayment Probability"],
                marker="o",
                label=strategy,
            )

        ax.set_title(f"{regime} Salary Growth")
        ax.set_xlabel("Initial Loan Balance (£000s)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Probability of Repayment Before Write-Off (%)")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8,
    )

    fig.suptitle("Sensitivity of Loan Repayment Probability to Initial Debt")

    plt.tight_layout(rect=[0, 0.15, 1, 0.93])
    plt.show()

    return stats
