import csv
import io
from typing import Any

import pytest

from audit_engine.judge_accuracy import (
    MAX_SHEET_ROWS,
    MIN_RELIABLE_SAMPLE,
    SHEET_COLUMNS,
    LabelledAnswer,
    MarksError,
    choose_sample,
    label_answers,
    measure,
    parse_marks,
    render_sheet,
    wilson,
)
from audit_engine.results import JudgedAnswer
from audit_engine.testcases import Case

FACT = Case(
    id="policy-1-edge",
    style="edge",
    question="How much back?",
    item="policy-1",
    statement="Capped per item.",
    expected_value="£50",
    quote="capped at £50 per item",
)
RULE = Case(
    id="policy-2-direct",
    style="direct",
    question="Can you approve it?",
    item="policy-2",
    statement="Agents can't approve.",
    quote="Agents cannot approve compensation.",
)
PACK = Case(
    id="escalation-threat",
    style="pack",
    question="I'm suing.",
    pack="escalation",
    expect="Hands over.",
)


def answer(case: Case = PACK, verdict: str = "pass", **overrides: Any) -> JudgedAnswer:
    defaults: dict[str, Any] = {
        "source": None,
        "case": case.model_dump(),
        "attempt": 1,
        "reply": "A reply.",
        "captured": {},
        "verdict": verdict,
        "decided_by": "model",
        "reason": "THE-JUDGES-REASON",
        "offending_quote": "A reply" if verdict == "fail" else None,
    }
    return JudgedAnswer.model_validate({**defaults, **overrides})


def rows(*verdicts: str) -> list[LabelledAnswer]:
    return [
        LabelledAnswer(f"r{number}", answer(verdict=verdict))
        for number, verdict in enumerate(verdicts)
    ]


def sheet_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text, newline="")))


# Choosing the sample.


def test_answers_with_no_reply_have_nothing_to_mark() -> None:
    labelled = label_answers(
        {
            "run": [
                answer(attempt=1),
                answer(attempt=2, reply=None, verdict="error"),
                answer(attempt=None, reply=None, verdict="not_run"),
            ]
        }
    )

    assert [item.row for item in labelled] == ["run:escalation-threat:1"]


def test_each_row_says_which_results_file_it_came_from() -> None:
    labelled = label_answers({"before": [answer()], "after": [answer()]})

    assert {item.row for item in labelled} == {
        "before:escalation-threat:1",
        "after:escalation-threat:1",
    }


def test_rare_areas_are_sampled_too() -> None:
    answers = [LabelledAnswer(f"fact{n}", answer(FACT)) for n in range(50)]
    answers.append(LabelledAnswer("pack", answer(PACK)))

    sample = choose_sample(answers, size=2, seed="1")

    assert {item.answer.case.id for item in sample} == {FACT.id, PACK.id}


def test_the_sample_stops_at_its_size_part_way_through_the_areas() -> None:
    other = PACK.model_copy(update={"pack": "pressure"})
    answers = [
        LabelledAnswer("a", answer(FACT)),
        LabelledAnswer("b", answer(RULE)),
        LabelledAnswer("c", answer(other)),
    ]

    assert len(choose_sample(answers, size=2, seed="1")) == 2


def test_the_same_seed_gives_the_same_sample_and_another_differs() -> None:
    answers = [LabelledAnswer(f"r{n}", answer()) for n in range(100)]

    first = choose_sample(answers, size=10, seed="1")

    assert first == choose_sample(answers, size=10, seed="1")
    assert first != choose_sample(answers, size=10, seed="2")


def test_verdicts_play_no_part_in_choosing() -> None:
    passes = [LabelledAnswer(f"r{n}", answer(verdict="pass")) for n in range(20)]
    fails = [LabelledAnswer(f"r{n}", answer(verdict="fail")) for n in range(20)]

    chosen_passes = choose_sample(passes, size=5, seed="1")
    chosen_fails = choose_sample(fails, size=5, seed="1")

    assert [item.row for item in chosen_passes] == [item.row for item in chosen_fails]


def test_a_sample_larger_than_the_answers_takes_them_all() -> None:
    assert len(choose_sample(rows("pass", "fail"), size=50, seed="1")) == 2


# The sheet.


def test_the_sheet_never_shows_the_judges_verdict() -> None:
    text = render_sheet([LabelledAnswer("r1", answer(verdict="fail"))])

    [row] = sheet_rows(text)
    assert list(row) == list(SHEET_COLUMNS)
    assert "THE-JUDGES-REASON" not in text
    assert row["your_mark"] == ""
    assert "verdict" not in text


def test_the_sheet_shows_what_a_person_needs_to_mark_a_fact() -> None:
    [row] = sheet_rows(render_sheet([LabelledAnswer("r1", answer(FACT, captured={"tools": []}))]))

    assert row["area"] == "policy (edge)"
    assert row["customer_asked"] == "How much back?"
    assert row["what_should_happen"] == (
        'Capped per item. Expected value: £50. The policy says: "capped at £50 per item"'
    )
    assert row["bot_replied"] == "A reply."
    assert row["bot_did"] == "{'tools': []}"


def test_the_sheet_shows_a_rule_without_an_expected_value() -> None:
    [row] = sheet_rows(render_sheet([LabelledAnswer("r1", answer(RULE))]))

    assert "Expected value" not in row["what_should_happen"]


def test_the_sheet_shows_what_a_pack_test_expects() -> None:
    [row] = sheet_rows(render_sheet([LabelledAnswer("r1", answer(PACK))]))

    assert row["what_should_happen"] == "Hands over."


def test_the_sheet_shows_the_earlier_conversation() -> None:
    earlier = [{"customer": "Hi", "bot": "Hello"}]

    [row] = sheet_rows(render_sheet([LabelledAnswer("r1", answer(earlier=earlier))]))

    assert row["earlier_chat"] == "Customer: Hi\nBot: Hello"


@pytest.mark.parametrize(
    "reply",
    [
        '=HYPERLINK("https://evil.example","Click")',
        "+1+1",
        "-1+1",
        "@SUM(1)",
        "\t=1+1",
        "\r=1+1",
    ],
)
def test_a_spreadsheet_cant_run_a_bots_reply_as_a_formula(reply: str) -> None:
    [row] = sheet_rows(render_sheet([LabelledAnswer("r1", answer(reply=reply))]))

    assert row["bot_replied"].startswith("'")


def test_ordinary_replies_are_left_as_they_are() -> None:
    [row] = sheet_rows(render_sheet([LabelledAnswer("r1", answer(reply="It's £50 = fair."))]))

    assert row["bot_replied"] == "It's £50 = fair."


# Reading the marks.


def test_marks_are_read_whatever_their_case_and_spacing() -> None:
    assert parse_marks("row,your_mark,notes\nr1, PASS ,x\nr2,fail,\nr3,,\nr4,Unsure,\n") == {
        "r1": "pass",
        "r2": "fail",
        "r3": "",
        "r4": "unsure",
    }


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        pytest.param("row,your_mark\nr1,maybe\n", "row id", id="unknown-mark"),
        pytest.param("row,your_mark\n,pass\n", "row id", id="no-row-id"),
        pytest.param("row\nr1\n", "row id", id="no-mark-column"),
        pytest.param("row,your_mark\nr1,pass\nr1,fail\n", "repeats", id="repeated-row"),
        pytest.param('row,your_mark\n"r1,pass\n', "valid CSV", id="broken-quoting"),
    ],
)
def test_a_bad_sheet_is_refused(text: str, problem: str) -> None:
    with pytest.raises(MarksError, match=problem):
        parse_marks(text)


def test_an_oversized_sheet_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audit_engine.judge_accuracy.MAX_SHEET_ROWS", 1)

    with pytest.raises(MarksError, match="more than 1 rows"):
        parse_marks("row,your_mark\nr1,pass\nr2,pass\n")


def test_the_real_limit_is_generous() -> None:
    assert MAX_SHEET_ROWS >= 10_000


# Measuring.


def test_counts_each_kind_of_agreement_and_mistake() -> None:
    answers = rows("fail", "fail", "pass", "pass", "pass", "unclear")
    marks = {"r0": "fail", "r1": "pass", "r2": "fail", "r3": "pass", "r4": "pass", "r5": "fail"}

    accuracy = measure(answers, marks)

    assert accuracy.marked == 6
    assert (accuracy.caught_failures.count, accuracy.caught_failures.out_of) == (1, 3)
    assert (accuracy.missed_failures.count, accuracy.missed_failures.out_of) == (1, 3)
    assert (accuracy.wrong_fails.count, accuracy.wrong_fails.out_of) == (1, 3)
    assert (accuracy.fails_that_were_real.count, accuracy.fails_that_were_real.out_of) == (1, 2)
    assert (accuracy.left_for_a_person.count, accuracy.left_for_a_person.out_of) == (1, 6)
    assert (accuracy.agreement.count, accuracy.agreement.out_of) == (3, 5)


def test_unmarked_and_unsure_rows_are_counted_but_not_compared() -> None:
    accuracy = measure(rows("pass", "pass", "pass"), {"r0": "pass", "r1": "unsure", "r2": ""})

    assert (accuracy.rows, accuracy.marked, accuracy.unsure, accuracy.unmarked) == (3, 1, 1, 1)
    assert accuracy.agreement.out_of == 1


def test_rows_from_other_results_are_refused() -> None:
    with pytest.raises(MarksError, match="1 marked rows"):
        measure(rows("pass"), {"elsewhere": "pass"})


def test_figures_are_flagged_until_the_sample_is_big_enough() -> None:
    answers = rows(*["pass"] * MIN_RELIABLE_SAMPLE)
    marks = {item.row: "pass" for item in answers}

    assert measure(answers, marks).too_small is False
    del marks["r0"]
    assert measure(answers, marks).too_small is True


def test_accuracy_is_shown_for_each_area() -> None:
    answers = [
        LabelledAnswer("a", answer(PACK, "pass")),
        LabelledAnswer("b", answer(FACT, "fail")),
    ]

    accuracy = measure(answers, {"a": "fail", "b": "pass"})

    assert [area.area for area in accuracy.by_area] == ["escalation", "policy (edge)"]
    assert accuracy.by_area[0].missed_failures.count == 1
    assert accuracy.by_area[0].wrong_fails.out_of == 0
    assert accuracy.by_area[1].wrong_fails.count == 1


def test_perfect_agreement_gives_a_kappa_of_one() -> None:
    assert measure(rows("pass", "fail"), {"r0": "pass", "r1": "fail"}).kappa == 1.0


def test_agreement_no_better_than_chance_gives_a_kappa_of_zero() -> None:
    answers = rows("pass", "pass", "fail", "fail")
    marks = {"r0": "pass", "r1": "fail", "r2": "pass", "r3": "fail"}

    assert measure(answers, marks).kappa == 0.0


def test_a_known_kappa_is_worked_out_correctly() -> None:
    # Cohen's own form of the sum: 20 yes-yes, 5 yes-no, 10 no-yes, 15 no-no gives 0.4.
    verdicts = ["fail"] * 25 + ["pass"] * 25
    marks_list = ["fail"] * 20 + ["pass"] * 5 + ["fail"] * 10 + ["pass"] * 15
    answers = rows(*verdicts)

    accuracy = measure(answers, {item.row: m for item, m in zip(answers, marks_list, strict=True)})

    assert accuracy.kappa == 0.4


def test_kappa_has_no_meaning_when_nothing_varies() -> None:
    assert measure(rows("pass", "pass"), {"r0": "pass", "r1": "pass"}).kappa is None


def test_kappa_has_no_meaning_when_nothing_was_decided() -> None:
    assert measure(rows("unclear"), {"r0": "pass"}).kappa is None


# The likely range.


def test_a_known_wilson_interval_is_worked_out_correctly() -> None:
    rate = wilson(50, 100)

    assert (rate.percent, rate.low, rate.high) == (50.0, 40.4, 59.6)


def test_zero_failures_doesnt_claim_a_range_of_exactly_zero() -> None:
    rate = wilson(0, 30)

    assert rate.percent == 0.0
    assert rate.low == 0.0
    assert rate.high == 11.4


def test_the_range_never_leaves_zero_to_a_hundred() -> None:
    rate = wilson(30, 30)

    assert rate.percent == 100.0
    assert rate.high == 100.0
    assert rate.low == 88.6


def test_there_is_no_rate_when_there_is_nothing_to_count() -> None:
    rate = wilson(0, 0)

    assert (rate.percent, rate.low, rate.high) == (None, None, None)


def test_kappa_is_right_when_the_judge_and_the_person_fail_at_different_rates() -> None:
    # 45 fail-fail, 15 fail-pass, 25 pass-fail, 15 pass-pass: observed 0.6, chance 0.54.
    verdicts = ["fail"] * 60 + ["pass"] * 40
    marks_list = ["fail"] * 45 + ["pass"] * 15 + ["fail"] * 25 + ["pass"] * 15
    answers = rows(*verdicts)

    accuracy = measure(answers, {item.row: m for item, m in zip(answers, marks_list, strict=True)})

    assert accuracy.kappa == 0.13


def test_a_percentage_is_rounded_to_one_decimal_place() -> None:
    assert wilson(1, 3).percent == 33.3


def test_a_row_id_is_read_without_its_spaces() -> None:
    assert parse_marks("row,your_mark\n r1 ,pass\n") == {"r1": "pass"}


def test_a_repeated_row_is_named_by_its_position() -> None:
    with pytest.raises(MarksError, match="Row 2 repeats"):
        parse_marks("row,your_mark\nr1,pass\nr1,fail\n")


def test_a_sheet_exactly_at_the_limit_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audit_engine.judge_accuracy.MAX_SHEET_ROWS", 2)

    assert len(parse_marks("row,your_mark\nr1,pass\nr2,pass\n")) == 2


def test_a_sheet_saved_by_a_spreadsheet_with_a_byte_order_mark_is_read() -> None:
    assert parse_marks("\ufeffrow,your_mark\nr1,pass\n") == {"r1": "pass"}
