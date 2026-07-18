from risk.limits import PortfolioExposure, gate_new_order


def _flat_portfolio():
    return PortfolioExposure(net_delta=0.0, net_vega=0.0, beta_weighted_net_delta=0.0)


def test_order_allowed_when_within_all_limits():
    result = gate_new_order(_flat_portfolio(), candidate_delta=50, candidate_vega=100, candidate_beta_weighted_delta=50)
    assert result.allowed is True
    assert result.reasons == []


def test_order_blocked_on_delta_breach():
    result = gate_new_order(_flat_portfolio(), candidate_delta=600, candidate_vega=0, candidate_beta_weighted_delta=0)
    assert result.allowed is False
    assert any("net_delta" in r for r in result.reasons)


def test_order_blocked_on_vega_breach():
    result = gate_new_order(_flat_portfolio(), candidate_delta=0, candidate_vega=2500, candidate_beta_weighted_delta=0)
    assert result.allowed is False
    assert any("net_vega" in r for r in result.reasons)


def test_order_blocked_on_beta_weighted_delta_breach():
    result = gate_new_order(_flat_portfolio(), candidate_delta=0, candidate_vega=0, candidate_beta_weighted_delta=800)
    assert result.allowed is False
    assert any("beta_weighted_net_delta" in r for r in result.reasons)


def test_order_blocked_reports_multiple_reasons():
    result = gate_new_order(_flat_portfolio(), candidate_delta=600, candidate_vega=2500, candidate_beta_weighted_delta=800)
    assert result.allowed is False
    assert len(result.reasons) == 3


def test_gate_accounts_for_existing_portfolio_exposure():
    existing = PortfolioExposure(net_delta=480, net_vega=0, beta_weighted_net_delta=0)
    result = gate_new_order(existing, candidate_delta=50, candidate_vega=0, candidate_beta_weighted_delta=0)
    assert result.allowed is False
    assert result.resulting_exposure.net_delta == 530
