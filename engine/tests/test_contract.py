import time
from typing import Any

import pytest
from pydantic import ValidationError

from audit_engine.contract import (
    MAX_CONTRACT_CHARS,
    MAX_ITEMS,
    Contract,
    ContractError,
    Item,
    Source,
    parse_contract,
    render_contract,
)

SHA256 = "a" * 64


def fact(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "policy-1",
        "kind": "fact",
        "statement": "Compensation is capped per item.",
        "value": "£50",
        "section": "Lost or damaged items",
        "quote": "capped at £50 per item",
        **overrides,
    }


def rule(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "policy-2",
        "kind": "rule",
        "statement": "Agents must not approve compensation in chat.",
        "section": "How to claim",
        "quote": "Support agents cannot approve compensation in chat.",
        **overrides,
    }


def contract(*items: dict[str, Any]) -> Contract:
    return Contract.model_validate(
        {"source": {"document": "policy.md", "sha256": SHA256}, "items": list(items)}
    )


def test_renders_in_the_documented_format() -> None:
    assert render_contract(contract(fact(), rule())) == (
        "source:\n"
        "  document: policy.md\n"
        f"  sha256: {SHA256}\n"
        "items:\n"
        "- id: policy-1\n"
        "  kind: fact\n"
        "  statement: Compensation is capped per item.\n"
        "  value: £50\n"
        "  section: Lost or damaged items\n"
        "  quote: capped at £50 per item\n"
        "  approved: false\n"
        "- id: policy-2\n"
        "  kind: rule\n"
        "  statement: Agents must not approve compensation in chat.\n"
        "  section: How to claim\n"
        "  quote: Support agents cannot approve compensation in chat.\n"
        "  approved: false\n"
    )


def test_a_person_can_approve_an_item_by_editing_the_file() -> None:
    text = render_contract(contract(fact()))

    edited = parse_contract(text.replace("approved: false", "approved: true"))

    assert edited.items[0].approved is True


def test_a_contract_with_no_items_is_valid() -> None:
    assert parse_contract(render_contract(contract())).items == []


# Saved text must come back identical: quotes are checked word for word.


@pytest.mark.parametrize(
    "text",
    [
        "£50",
        "null",
        "yes",
        "no",
        "1.0",
        "50",
        "2026-10-30",
        "~",
        ": colon first",
        "ends with colon:",
        "# looks like a comment",
        "- looks like a list",
        "{looks: like a map}",
        "[looks, like, a list]",
        "*alias",
        "&anchor",
        "!tag",
        "| block",
        "'single' and \"double\" quotes",
        "  leading and trailing spaces  ",
        "two\nlines",
        "ends with a line break\n",
        "tab\tinside",
        "配送 доставка café 😀",
        "next line \N{NEXT LINE} and line separator \N{LINE SEPARATOR} inside",
        "bell \x07 and null \x00 inside",
        "x " * 1_000,
    ],
)
def test_text_survives_saving_and_loading_exactly(text: str) -> None:
    original = contract(
        fact(quote=text, value=text[:200], statement=text[:1_000], section=text[:300])
    )

    assert parse_contract(render_contract(original)) == original


def test_values_are_kept_as_text_not_turned_into_numbers() -> None:
    reloaded = parse_contract(render_contract(contract(fact(value="50"))))

    assert reloaded.items[0].value == "50"


# Items that make no sense must be refused.


@pytest.mark.parametrize(
    ("item", "problem"),
    [
        pytest.param(fact(value=None), "A fact must have a value", id="fact-without-value"),
        pytest.param(rule(value="£50"), "A rule must not have a value", id="rule-with-value"),
        pytest.param(fact(kind="opinion"), "kind", id="unknown-kind"),
        pytest.param(fact(id=""), "id", id="empty-id"),
        pytest.param(fact(id="two words"), "id", id="id-with-space"),
        pytest.param(fact(id="policy-1\n"), "id", id="id-with-line-break"),
        pytest.param(fact(id="x" * 301), "id", id="long-id"),
        pytest.param(fact(statement=""), "statement", id="empty-statement"),
        pytest.param(fact(statement="   "), "statement", id="blank-statement"),
        pytest.param(fact(statement="x" * 1_001), "statement", id="long-statement"),
        pytest.param(fact(value="  "), "value", id="blank-value"),
        pytest.param(fact(value="x" * 201), "value", id="long-value"),
        pytest.param(fact(value=50), "value", id="number-value"),
        pytest.param(fact(section=" "), "section", id="blank-section"),
        pytest.param(fact(section="x" * 301), "section", id="long-section"),
        pytest.param(fact(quote=""), "quote", id="empty-quote"),
        pytest.param(fact(quote="\n"), "quote", id="blank-quote"),
        pytest.param(fact(quote="x" * 4_001), "quote", id="long-quote"),
        pytest.param(fact(quote="broken \ud800 text"), "quote", id="broken-unicode"),
        pytest.param(fact(approved="yes"), "approved", id="approval-as-text"),
        pytest.param(fact(approved=1), "approved", id="approval-as-number"),
        pytest.param(fact(approved=None), "approved", id="approval-null"),
        pytest.param(fact(verified=True), "verified", id="unknown-field"),
    ],
)
def test_bad_items_are_refused(item: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValidationError, match=problem):
        Item.model_validate(item)


@pytest.mark.parametrize("missing", ["id", "kind", "statement", "section", "quote"])
def test_required_item_fields(missing: str) -> None:
    item = fact()
    del item[missing]

    with pytest.raises(ValidationError, match=missing):
        Item.model_validate(item)


def test_items_are_unapproved_until_someone_approves_them() -> None:
    assert Item.model_validate(fact()).approved is False


@pytest.mark.parametrize(
    "source",
    [
        pytest.param({"document": "", "sha256": SHA256}, id="empty-document"),
        pytest.param({"document": "../secrets.md", "sha256": SHA256}, id="path-traversal"),
        pytest.param({"document": "a\\b.md", "sha256": SHA256}, id="windows-path"),
        pytest.param({"document": "x" * 301, "sha256": SHA256}, id="long-document"),
        pytest.param({"document": "p.md", "sha256": "abc"}, id="short-fingerprint"),
        pytest.param({"document": "p.md", "sha256": "A" * 64}, id="upper-case-fingerprint"),
        pytest.param({"document": "p.md", "sha256": "g" * 64}, id="not-hex"),
        pytest.param({"document": "p.md", "sha256": SHA256 + "\n"}, id="line-break"),
        pytest.param({"document": "p.md"}, id="missing-fingerprint"),
        pytest.param({"document": "p.md", "sha256": SHA256, "url": "x"}, id="unknown-field"),
    ],
)
def test_bad_sources_are_refused(source: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Source.model_validate(source)


def test_item_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        contract(fact(), fact())


def test_too_many_items_are_refused() -> None:
    items = [fact(id=f"policy-{number}") for number in range(MAX_ITEMS + 1)]

    with pytest.raises(ValidationError, match="items"):
        contract(*items)


# Files come from people and other tools, so reading them must fail cleanly.


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        pytest.param("", "contract", id="empty"),
        pytest.param("just text", "contract", id="plain-text"),
        pytest.param("- a\n- list", "contract", id="list"),
        pytest.param("source: [", "not valid YAML", id="broken-yaml"),
        pytest.param("\tsource: x", "not valid YAML", id="tab-indent"),
        pytest.param("items: []", "source", id="missing-source"),
        pytest.param(f"source: {{document: p.md, sha256: {SHA256}}}", "items", id="missing-items"),
        pytest.param("!!python/object/apply:os.system [ls]", "not valid YAML", id="code-execution"),
        pytest.param("a: !!python/name:os.system", "not valid YAML", id="python-tag"),
        pytest.param("a: &a [*a]", "a", id="self-reference"),
        pytest.param("[" * 100_000, "not valid YAML", id="deep-nesting"),
        pytest.param("source: 1\nsource: 2", "repeats a key", id="duplicate-key"),
    ],
)
def test_bad_files_fail_with_a_contract_error(text: str, problem: str) -> None:
    with pytest.raises(ContractError, match=problem):
        parse_contract(text)


def test_a_second_approval_line_cannot_overrule_the_first() -> None:
    text = render_contract(contract(fact()))
    duplicated = text.replace("approved: false", "approved: false\n  approved: true")

    with pytest.raises(ContractError, match="repeats a key"):
        parse_contract(duplicated)


def test_merge_keys_cannot_slip_in_an_approval() -> None:
    text = (
        f"source: {{document: p.md, sha256: {SHA256}}}\n"
        "items:\n"
        "- id: a\n  kind: rule\n  statement: s\n  section: s\n  quote: q\n"
        "  <<: {approved: true}\n"
    )

    with pytest.raises(ContractError, match="not valid YAML"):
        parse_contract(text)


def test_alias_expansion_bombs_are_refused_quickly() -> None:
    bomb = "a: &a [x, x]\n" + "".join(
        f"{chr(98 + n)}: &{chr(98 + n)} [{f'*{chr(97 + n)}, ' * 9}*{chr(97 + n)}]\n"
        for n in range(20)
    )
    started = time.monotonic()

    with pytest.raises(ContractError):
        parse_contract(bomb)

    assert time.monotonic() - started < 5


def test_oversized_files_are_refused_before_reading() -> None:
    with pytest.raises(ContractError, match="longer than"):
        parse_contract("x" * (MAX_CONTRACT_CHARS + 1))


def test_errors_never_quote_the_file() -> None:
    text = render_contract(contract(fact(quote="private customer detail")))
    broken = text.replace("kind: fact", "kind: opinion")

    with pytest.raises(ContractError) as caught:
        parse_contract(broken)

    assert "private customer detail" not in str(caught.value)
    assert "items.0.kind" in str(caught.value)


def test_yaml_errors_never_quote_the_file() -> None:
    with pytest.raises(ContractError) as caught:
        parse_contract("quote: private customer detail\n  broken: [")

    assert "private customer detail" not in str(caught.value)


# Which rules forbid promises to act on an unverified approval: a reviewer's decision.


@pytest.mark.parametrize("decision", [True, False])
def test_a_reviewer_can_say_whether_a_rule_forbids_approval_promises(decision: bool) -> None:
    saved = contract(rule(forbids_approval_promises=decision))

    assert parse_contract(render_contract(saved)).items[0].forbids_approval_promises is decision


def test_a_rule_left_undecided_stays_undecided_and_older_contracts_still_load() -> None:
    saved = contract(rule())

    assert "forbids_approval_promises" not in render_contract(saved)
    assert saved.items[0].forbids_approval_promises is None


@pytest.mark.parametrize(
    ("item", "problem"),
    [
        pytest.param(fact(forbids_approval_promises=True), "Only a rule", id="on-a-fact"),
        pytest.param(fact(forbids_approval_promises=False), "Only a rule", id="false-on-a-fact"),
        pytest.param(rule(forbids_approval_promises="yes"), "forbids", id="as-text"),
        pytest.param(rule(forbids_approval_promises=1), "forbids", id="as-number"),
    ],
)
def test_bad_approval_decisions_are_refused(item: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValidationError, match=problem):
        Item.model_validate(item)
