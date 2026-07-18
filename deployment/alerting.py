"""
Alerting on unhandled exceptions and risk-limit breaches, via a Slack
incoming webhook. The webhook URL is optional (ALERT_SLACK_WEBHOOK_URL in
.env) -- if it isn't configured, alerts degrade to a log line rather than
failing the pipeline, since a research pipeline shouldn't crash because
alerting itself isn't set up yet.
"""
import logging
import os
import traceback

import requests

from deployment.logging_setup import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

_TIMEOUT_SECONDS = 10
_ALERT_WEBHOOK_ENV_VAR = "ALERT_SLACK_WEBHOOK_URL"


def send_alert(message: str) -> None:
    webhook_url = os.environ.get(_ALERT_WEBHOOK_ENV_VAR)
    if not webhook_url:
        logger.warning("ALERT (no Slack webhook configured): %s", message)
        return
    try:
        requests.post(webhook_url, json={"text": message}, timeout=_TIMEOUT_SECONDS)
    except requests.RequestException:
        logger.exception("Failed to deliver alert to Slack webhook")


def alert_unhandled_exception(exc: BaseException) -> None:
    send_alert(f"Unhandled exception in pipeline run:\n```{traceback.format_exc()}```")


def alert_risk_limit_breach(reasons: list[str]) -> None:
    send_alert("Risk limit breach blocked an order:\n" + "\n".join(f"- {r}" for r in reasons))
