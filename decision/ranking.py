"""
Ranks candidate contracts by expected P&L per unit of capital, after
excluding anything that failed a hard gate (factors/liquidity.py always;
factors/beta_regime.py during a crisis regime) -- gated candidates never
appear in the ranked output at all, per the spec, rather than being
included with a warning.
"""
from dataclasses import dataclass

from decision.edge import (
    Candidate,
    edge_per_contract,
    expected_pnl_per_capital,
    model_minus_market_edge,
)


@dataclass
class RankedCandidate:
    candidate: Candidate
    model_minus_market_edge: float
    edge_per_contract: float
    expected_pnl_per_capital: float
    contributing_factors: dict[str, float]


def passes_all_gates(candidate: Candidate) -> bool:
    return all(fr.passed_gate for fr in candidate.factor_results)


def rank_candidates(candidates: list[Candidate]) -> list[RankedCandidate]:
    ranked = []
    for candidate in candidates:
        if not passes_all_gates(candidate):
            continue
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                model_minus_market_edge=model_minus_market_edge(candidate),
                edge_per_contract=edge_per_contract(candidate),
                expected_pnl_per_capital=expected_pnl_per_capital(candidate),
                contributing_factors={
                    fr.factor_name: fr.score for fr in candidate.factor_results
                },
            )
        )
    ranked.sort(key=lambda r: r.expected_pnl_per_capital, reverse=True)
    return ranked


def to_audit_rows(ranked: list[RankedCandidate]) -> list[dict]:
    """Flat, auditable rows suitable for the dashboard/logging: ticker,
    strike, expiration, strategy, model edge, expected P&L per capital, and
    every factor score that contributed."""
    return [
        {
            "ticker": r.candidate.ticker,
            "strategy": r.candidate.strategy,
            "option_type": r.candidate.option_type,
            "strike": r.candidate.strike,
            "expiration": r.candidate.expiration.isoformat(),
            "dte": r.candidate.dte,
            "model_minus_market_edge": r.model_minus_market_edge,
            "edge_per_contract": r.edge_per_contract,
            "expected_pnl_per_capital": r.expected_pnl_per_capital,
            "contributing_factors": r.contributing_factors,
        }
        for r in ranked
    ]
