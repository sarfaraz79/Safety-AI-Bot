import json

import pytest

from demo_bot.tools import TOOLS

VALID_REFUND = '{"order_id": "EP-100245", "amount_gbp": 12.5, "reason": "Late delivery"}'


def test_refund_is_simulated_and_returns_a_reference() -> None:
    result = TOOLS["issue_refund"].run(VALID_REFUND)
    output = json.loads(result.output)

    assert result.ok
    assert output["status"] == "approved"
    assert output["amount_gbp"] == 12.5
    assert output["refund_reference"].startswith("RF-")


def test_escalation_returns_a_ticket_reference() -> None:
    result = TOOLS["escalate_to_human"].run('{"reason": "Customer asked for a person"}')

    assert result.ok
    assert json.loads(result.output)["ticket_reference"].startswith("HD-")


@pytest.mark.parametrize(
    ("raw_arguments", "problem"),
    [
        ("not json", "Invalid JSON"),
        ('{"order_id": "12345", "amount_gbp": 5, "reason": "x"}', "order_id"),
        ('{"order_id": "EP-100245", "amount_gbp": -5, "reason": "x"}', "amount_gbp"),
        ('{"order_id": "EP-100245", "amount_gbp": 99999, "reason": "x"}', "amount_gbp"),
        ('{"order_id": "EP-100245", "amount_gbp": 5}', "reason"),
        ('{"order_id": "EP-100245", "amount_gbp": 5, "reason": "x", "admin": true}', "admin"),
    ],
)
def test_invalid_arguments_are_reported_back_to_the_model(raw_arguments: str, problem: str) -> None:
    result = TOOLS["issue_refund"].run(raw_arguments)

    assert not result.ok
    assert problem in json.loads(result.output)["error"]


def test_specs_describe_arguments_as_strict_json_schema() -> None:
    spec = TOOLS["issue_refund"].spec

    assert spec.name == "issue_refund"
    assert spec.parameters["required"] == ["order_id", "amount_gbp", "reason"]
    assert spec.parameters["additionalProperties"] is False
