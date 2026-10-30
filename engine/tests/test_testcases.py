from typing import Any

import pytest
from pydantic import ValidationError

from audit_engine.contract import Source
from audit_engine.testcases import (
    MAX_CASES,
    MAX_QUESTION_CHARS,
    MAX_TESTS_FILE_CHARS,
    Case,
    CaseFile,
    CaseFileError,
    parse_cases,
    render_cases,
)

SOURCE = Source(document="policy.md", sha256="a" * 64)


def item_case(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "policy-1-reworded",
        "style": "reworded",
        "question": "My vase arrived smashed. What's the most you'll give me back?",
        "item": "policy-1",
        "statement": "Compensation is capped per item.",
        "expected_value": "£50",
        "quote": "capped at £50 per item",
        **overrides,
    }


def pack_case(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "escalation-legal-threat",
        "style": "pack",
        "question": "I'm taking you to court.",
        "pack": "escalation",
        "expect": "Hands the conversation to a person.",
        **overrides,
    }


def case_file(*cases: dict[str, Any]) -> CaseFile:
    return CaseFile.model_validate({"source": SOURCE.model_dump(), "tests": list(cases)})


def test_renders_in_reading_order_without_empty_fields() -> None:
    text = render_cases(case_file(item_case()))

    assert text == (
        "source:\n"
        "  document: policy.md\n"
        f"  sha256: {'a' * 64}\n"
        "tests:\n"
        "- id: policy-1-reworded\n"
        "  style: reworded\n"
        "  question: My vase arrived smashed. What's the most you'll give me back?\n"
        "  item: policy-1\n"
        "  statement: Compensation is capped per item.\n"
        "  expected_value: £50\n"
        "  quote: capped at £50 per item\n"
    )


@pytest.mark.parametrize(
    "question",
    [
        "yes",
        "null",
        "50",
        "# comment",
        "two\nlines",
        "  spaced  ",
        '"quoted" and {{braces}}',
        "配送 😀",
        "bell \x07 inside",
    ],
)
def test_questions_survive_saving_and_loading_exactly(question: str) -> None:
    original = case_file(item_case(question=question))

    assert parse_cases(render_cases(original)) == original


def test_a_file_can_mix_item_and_pack_tests() -> None:
    original = case_file(item_case(), pack_case())

    assert parse_cases(render_cases(original)) == original


def test_a_file_of_only_pack_tests_needs_no_source() -> None:
    original = CaseFile.model_validate({"tests": [pack_case()]})

    assert parse_cases(render_cases(original)).source is None


def test_a_rule_has_no_expected_value() -> None:
    assert Case.model_validate(item_case(expected_value=None)).expected_value is None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(item_case(id=""), id="empty-id"),
        pytest.param(item_case(id="two words"), id="id-with-space"),
        pytest.param(item_case(id="a\n"), id="id-with-line-break"),
        pytest.param(item_case(style="exam"), id="unknown-style"),
        pytest.param(item_case(question=""), id="empty-question"),
        pytest.param(item_case(question="   "), id="blank-question"),
        pytest.param(item_case(question="x" * (MAX_QUESTION_CHARS + 1)), id="long-question"),
        pytest.param(item_case(question="broken \ud800"), id="broken-unicode"),
        pytest.param(item_case(item=None), id="item-test-without-item"),
        pytest.param(item_case(statement=None), id="item-test-without-statement"),
        pytest.param(item_case(quote=None), id="item-test-without-quote"),
        pytest.param(item_case(pack="escalation"), id="item-test-with-pack"),
        pytest.param(item_case(expect="Something."), id="item-test-with-expect"),
        pytest.param(item_case(style="pack"), id="pack-style-with-item-fields"),
        pytest.param(pack_case(pack=None), id="pack-test-without-pack"),
        pytest.param(pack_case(expect=None), id="pack-test-without-expect"),
        pytest.param(pack_case(item="policy-1"), id="pack-test-with-item"),
        pytest.param(pack_case(quote="q"), id="pack-test-with-quote"),
        pytest.param(pack_case(expected_value="£50"), id="pack-test-with-value"),
        pytest.param(pack_case(pack="Has Spaces"), id="bad-pack-name"),
        pytest.param(pack_case(unknown=True), id="unknown-field"),
    ],
)
def test_bad_tests_are_refused(case: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(case)


def test_a_question_exactly_at_the_limit_is_accepted() -> None:
    assert (
        len(Case.model_validate(item_case(question="x" * MAX_QUESTION_CHARS)).question)
        == MAX_QUESTION_CHARS
    )


def test_test_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        case_file(item_case(), item_case())


def test_too_many_tests_are_refused() -> None:
    cases = [item_case(id=f"t{number}") for number in range(MAX_CASES + 1)]

    with pytest.raises(ValidationError, match="tests"):
        case_file(*cases)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        pytest.param("", "test file", id="empty"),
        pytest.param("- a\n- list", "test file", id="list"),
        pytest.param("tests: [", "not valid YAML", id="broken-yaml"),
        pytest.param("!!python/object/apply:os.system [ls]", "not valid YAML", id="code-execution"),
        pytest.param("tests: []\ntests: []", "repeats a key", id="duplicate-key"),
        pytest.param("source: 5\ntests: []", "source", id="bad-source"),
        pytest.param("x" * (MAX_TESTS_FILE_CHARS + 1), "longer than", id="oversized"),
    ],
)
def test_bad_files_fail_with_a_clear_error(text: str, problem: str) -> None:
    with pytest.raises(CaseFileError, match=problem):
        parse_cases(text)


def test_errors_never_quote_the_file() -> None:
    text = render_cases(case_file(item_case(quote="private customer detail")))

    with pytest.raises(CaseFileError) as caught:
        parse_cases(text.replace("style: reworded", "style: exam"))

    assert "private customer detail" not in str(caught.value)
    assert "tests.0.style" in str(caught.value)
