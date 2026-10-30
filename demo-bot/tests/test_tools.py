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


def refund_args(**overrides: object) -> str:
    return json.dumps({"order_id": "EP-100245", "amount_gbp": 5, "reason": "x", **overrides})


@pytest.mark.parametrize(
    "raw_arguments",
    [
        pytest.param("", id="empty"),
        pytest.param("null", id="null"),
        pytest.param("[]", id="array"),
        pytest.param('"refund"', id="string"),
        pytest.param(refund_args(order_id="EP-١٠٠٢٤٥"), id="non-latin-digits"),
        pytest.param(refund_args(order_id="ep-100245"), id="lower-case-order"),
        pytest.param(refund_args(order_id="EP-1002456"), id="long-order"),
        pytest.param(refund_args(order_id="EP-100245\nEP-100246"), id="two-orders"),
        pytest.param(refund_args(amount_gbp=0), id="zero"),
        pytest.param(refund_args(amount_gbp=0.001), id="fraction-of-a-penny"),
        pytest.param(refund_args(amount_gbp=10_000.01), id="just-over-the-ceiling"),
        pytest.param(refund_args(amount_gbp="5"), id="amount-as-text"),
        pytest.param(refund_args(amount_gbp=True), id="amount-as-boolean"),
        pytest.param(refund_args(amount_gbp=None), id="amount-null"),
        pytest.param(refund_args().replace("5", "NaN"), id="not-a-number"),
        pytest.param(refund_args().replace("5", "Infinity"), id="infinity"),
        pytest.param(refund_args(reason=""), id="empty-reason"),
        pytest.param(refund_args(reason="   "), id="blank-reason"),
        pytest.param(refund_args(reason="x" * 501), id="long-reason"),
        pytest.param(refund_args().replace('"x"', '"\\ud800"'), id="broken-unicode-reason"),
    ],
)
def test_bad_refund_arguments_never_run_the_tool(raw_arguments: str) -> None:
    result = TOOLS["issue_refund"].run(raw_arguments)

    assert not result.ok
    assert "error" in json.loads(result.output)


@pytest.mark.parametrize(
    "raw_arguments",
    [
        refund_args(amount_gbp=0.01),
        refund_args(amount_gbp=7.99),
        refund_args(amount_gbp=10_000),
        refund_args(reason="x" * 500),
        refund_args(reason="Late £ 🚚 配送"),
        refund_args(reason="  padded  "),
    ],
)
def test_good_refund_arguments_run_the_tool(raw_arguments: str) -> None:
    assert TOOLS["issue_refund"].run(raw_arguments).ok


def test_errors_never_repeat_the_bad_value() -> None:
    result = TOOLS["issue_refund"].run(refund_args(order_id="SECRET-VALUE"))

    assert "SECRET-VALUE" not in result.output


def test_every_call_gets_its_own_reference() -> None:
    outputs = [json.loads(TOOLS["issue_refund"].run(VALID_REFUND).output) for _ in range(100)]

    assert len({output["refund_reference"] for output in outputs}) == 100


@pytest.mark.parametrize("reason", ["", "   ", "x" * 501])
def test_escalation_needs_a_real_reason(reason: str) -> None:
    assert not TOOLS["escalate_to_human"].run(json.dumps({"reason": reason})).ok


def test_specs_describe_arguments_as_strict_json_schema() -> None:
    spec = TOOLS["issue_refund"].spec

    assert spec.name == "issue_refund"
    assert spec.parameters["required"] == ["order_id", "amount_gbp", "reason"]
    assert spec.parameters["additionalProperties"] is False
