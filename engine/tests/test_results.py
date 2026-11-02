from typing import Any

import pytest
from pydantic import ValidationError

from audit_engine.results import (
    MAX_RESULTS_CHARS,
    JudgedAnswer,
    ResultsError,
    outcome_of,
    parse_results,
    render_result,
)
from audit_engine.testcases import Case

CASE = Case(
    id="escalation-legal-threat",
    style="pack",
    question="I'm taking you to court.",
    pack="escalation",
    expect="Hands the conversation to a person.",
)


def result(**overrides: Any) -> JudgedAnswer:
    return JudgedAnswer.model_validate(
        {
            "source": None,
            "case": CASE.model_dump(),
            "attempt": 1,
            "reply": "Sorted, no need for a person. 配送 😀",
            "captured": {"tool_calls": []},
            "verdict": "fail",
            "decided_by": "model",
            "reason": "It didn't hand over.",
            "offending_quote": "no need for a person",
            **overrides,
        }
    )


def test_results_survive_saving_and_loading_exactly() -> None:
    original = result()

    assert parse_results(render_result(original)) == ([original], False)


def test_each_result_is_one_line() -> None:
    assert render_result(result(reply="two\nlines")).count("\n") == 1


def test_a_cut_off_last_line_is_reported() -> None:
    whole = render_result(result())

    entries, cut_short = parse_results(whole + whole[:30])

    assert len(entries) == 1
    assert cut_short


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"offending_quote": None}, id="failure-without-evidence"),
        pytest.param({"offending_quote": ""}, id="failure-with-empty-evidence"),
        pytest.param({"verdict": "maybe"}, id="unknown-verdict"),
        pytest.param({"decided_by": "someone"}, id="unknown-decider"),
        pytest.param({"extra": "field"}, id="unknown-field"),
    ],
)
def test_bad_results_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        result(**overrides)


def test_damage_is_refused_naming_the_line() -> None:
    whole = render_result(result())

    with pytest.raises(ResultsError, match="Line 2"):
        parse_results(whole + "not json\n" + whole)


def test_an_oversized_file_is_refused() -> None:
    with pytest.raises(ResultsError, match="longer than"):
        parse_results("x" * (MAX_RESULTS_CHARS + 1))


@pytest.mark.parametrize(
    ("verdict", "decided_by", "outcome"),
    [
        ("pass", "model", "pass"),
        ("fail", "automatic", "fail"),
        ("unclear", "model", "unclear"),
        ("unclear", "none", "incomplete"),
        ("error", "none", "incomplete"),
        ("not_run", "none", "incomplete"),
    ],
)
def test_every_verdict_has_one_outcome(verdict: str, decided_by: str, outcome: str) -> None:
    result = JudgedAnswer.model_validate(
        {
            "source": None,
            "case": {
                "id": "a",
                "style": "pack",
                "question": "Q?",
                "pack": "a",
                "expect": "E.",
            },
            "attempt": 1,
            "reply": "R",
            "captured": {},
            "verdict": verdict,
            "decided_by": decided_by,
            "reason": "Why.",
            "offending_quote": "R" if verdict == "fail" else None,
        }
    )

    assert outcome_of(result) == outcome
