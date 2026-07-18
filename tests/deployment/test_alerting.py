from deployment import alerting


def test_send_alert_logs_warning_when_no_webhook_configured(monkeypatch, caplog):
    monkeypatch.delenv("ALERT_SLACK_WEBHOOK_URL", raising=False)
    with caplog.at_level("WARNING"):
        alerting.send_alert("something happened")
    assert any("ALERT" in record.message for record in caplog.records)


def test_send_alert_posts_to_webhook_when_configured(monkeypatch):
    monkeypatch.setenv("ALERT_SLACK_WEBHOOK_URL", "https://hooks.slack.example/test")
    calls = []
    monkeypatch.setattr(
        alerting.requests,
        "post",
        lambda url, json, timeout: calls.append((url, json)),
    )

    alerting.send_alert("risk limit breached")

    assert len(calls) == 1
    assert calls[0][0] == "https://hooks.slack.example/test"
    assert calls[0][1] == {"text": "risk limit breached"}


def test_alert_risk_limit_breach_formats_reasons(monkeypatch):
    monkeypatch.delenv("ALERT_SLACK_WEBHOOK_URL", raising=False)
    captured = {}
    monkeypatch.setattr(alerting, "send_alert", lambda msg: captured.setdefault("msg", msg))

    alerting.alert_risk_limit_breach(["net_delta too high", "net_vega too high"])

    assert "net_delta too high" in captured["msg"]
    assert "net_vega too high" in captured["msg"]


def test_alert_unhandled_exception_includes_traceback(monkeypatch):
    monkeypatch.delenv("ALERT_SLACK_WEBHOOK_URL", raising=False)
    captured = {}
    monkeypatch.setattr(alerting, "send_alert", lambda msg: captured.setdefault("msg", msg))

    try:
        raise ValueError("boom")
    except ValueError as exc:
        alerting.alert_unhandled_exception(exc)

    assert "ValueError" in captured["msg"]
    assert "boom" in captured["msg"]
