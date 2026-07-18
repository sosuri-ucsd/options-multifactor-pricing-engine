from risk.hedging import generate_hedge_order, needs_delta_hedge


def test_needs_delta_hedge_below_threshold():
    assert not needs_delta_hedge(50, threshold=100)


def test_needs_delta_hedge_above_threshold():
    assert needs_delta_hedge(150, threshold=100)


def test_generate_hedge_order_none_when_within_threshold():
    assert generate_hedge_order(50, threshold=100) is None


def test_generate_hedge_order_sells_when_delta_too_positive():
    order = generate_hedge_order(250, threshold=100)
    assert order == {"side": "sell", "shares": 250}


def test_generate_hedge_order_buys_when_delta_too_negative():
    order = generate_hedge_order(-250, threshold=100)
    assert order == {"side": "buy", "shares": 250}
