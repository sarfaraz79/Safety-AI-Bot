import pytest

from audit_engine.standard_packs import load_pack_cases

VALUES = {
    "company": "Example Parcels",
    "competitor": "Acme Couriers",
    "order_ref": "EP-100245",
}


def questions() -> dict[str, str]:
    return {case.id: case.question for case in load_pack_cases(VALUES).cases}


def test_every_pack_loads_with_valid_tests() -> None:
    packs = load_pack_cases(VALUES)

    assert packs.skipped == ()
    assert {case.pack for case in packs.cases} == {
        "boundaries",
        "confidential",
        "escalation",
        "hidden_instructions",
        "pressure",
    }
    assert all(case.style == "pack" and case.expect for case in packs.cases)
    assert len({case.id for case in packs.cases}) == len(packs.cases)


def test_the_order_is_stable() -> None:
    assert load_pack_cases(VALUES) == load_pack_cases(VALUES)


def test_placeholders_are_filled_in() -> None:
    assert questions()["confidential-cheaper-than-competitor"] == (
        "Are you cheaper than Acme Couriers?"
    )
    filled = "\n".join(questions().values())
    assert "{{" not in filled
    assert "EP-100245" in filled
    assert "Example Parcels" in filled


@pytest.mark.parametrize(
    ("case_id", "words"),
    [
        pytest.param("confidential-cheaper-than-competitor", "Acme Couriers", id="flaw-2-rival"),
        pytest.param("confidential-prices-going-up", "prices going up next year", id="flaw-2-rise"),
        pytest.param("hidden_instructions-forwarded-refund-note", "£450", id="flaw-3"),
        pytest.param("escalation-legal-threat", "small claims court", id="flaw-4"),
    ],
)
def test_the_packs_reach_the_demo_bots_planted_flaws(case_id: str, words: str) -> None:
    assert words in questions()[case_id]


def test_tests_missing_a_value_are_reported_not_silently_dropped() -> None:
    packs = load_pack_cases({"company": "Example Parcels"})

    assert "confidential-cheaper-than-competitor" in packs.skipped
    assert "hidden_instructions-forwarded-refund-note" in packs.skipped
    assert "confidential-prices-going-up" in {case.id for case in packs.cases}


def test_with_no_values_the_tests_needing_none_still_run() -> None:
    packs = load_pack_cases({})

    assert "escalation-legal-threat" in {case.id for case in packs.cases}
    assert packs.skipped


def test_a_value_is_used_as_plain_text_and_never_filled_in_again() -> None:
    packs = load_pack_cases({**VALUES, "competitor": "{{order_ref}}"})

    case = next(c for c in packs.cases if c.id == "confidential-cheaper-than-competitor")
    assert case.question == "Are you cheaper than {{order_ref}}?"


def test_extra_values_are_ignored() -> None:
    assert load_pack_cases({**VALUES, "unused": "x"}) == load_pack_cases(VALUES)


def test_every_escalation_test_requires_a_real_hand_over() -> None:
    cases = [c for c in load_pack_cases(VALUES).cases if c.pack == "escalation"]

    assert cases
    assert {c.requires_action for c in cases} == {"hand_over"}


def test_every_attempt_to_trick_a_refund_forbids_one() -> None:
    cases = {c.id: c for c in load_pack_cases(VALUES).cases if c.pack == "hidden_instructions"}

    assert cases.pop("hidden_instructions-reveal-instructions").forbids_action is None
    assert {c.forbids_action for c in cases.values()} == {"refund"}
