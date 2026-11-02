"""Keeps a judge evaluation honest: a frozen judge, unseen answers and two independent markers."""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from audit_engine.judge import JUDGE_ID
from audit_engine.judge_accuracy import (
    MIN_RELIABLE_SAMPLE,
    LabelledAnswer,
    MarkerRow,
    cohen_kappa,
    render_cells,
)
from audit_engine.results import JudgedAnswer

MAX_EVALUATION_FILE_CHARS = 100_000_000
MIN_MARKER_KAPPA = 0.70
MAX_NOTE_CHARS = 500

Purpose = Literal["tuning", "development", "evaluation"]

ADJUDICATION_COLUMNS = (
    "row",
    "area",
    "customer_asked",
    "earlier_chat",
    "what_should_happen",
    "bot_replied",
    "bot_did",
    "first_mark",
    "first_reason",
    "second_mark",
    "second_reason",
    "your_mark",
    "your_severity",
    "notes",
)


class EvaluationFileError(ValueError):
    """A freeze or ledger file can't be used."""


class JudgeFreeze(BaseModel):
    """Exactly which judge an evaluation is about."""

    model_config = ConfigDict(extra="forbid")

    judge_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=200)
    frozen_at: datetime


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: Purpose
    note: str = Field(max_length=MAX_NOTE_CHARS)
    recorded_at: datetime


class Ledger(BaseModel):
    """Every answer already seen, so it can't be passed off as unseen later."""

    model_config = ConfigDict(extra="forbid")

    answers: dict[str, LedgerEntry] = Field(default_factory=dict)


class MarkerAgreement(BaseModel):
    both_marked: int
    agreed: int
    percent: float | None
    kappa: float | None
    disagreements: list[str]
    marked_by_one_only: int


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    """What's been supplied to show an evaluation is fair. Missing evidence counts against it."""

    freeze: JudgeFreeze | None
    ledger: Ledger | None
    agreement: MarkerAgreement | None
    no_ai_statement: bool


class Independence(BaseModel):
    """`problems` says why these figures must not be described as an independent evaluation."""

    eligible: bool
    problems: list[str]


def answer_id(answer: JudgedAnswer) -> str:
    """Identify an answer by its words, so copying it to another file doesn't make it new."""
    earlier = [(turn.customer, turn.bot) for turn in answer.earlier]
    text = repr((answer.case.question, earlier, answer.reply))
    return hashlib.sha256(text.encode()).hexdigest()


def add_to_ledger(
    ledger: Ledger,
    answers: Iterable[JudgedAnswer],
    *,
    purpose: Purpose,
    note: str,
    recorded_at: datetime,
) -> int:
    """Record answers as seen, and return how many are new. An earlier record is never replaced."""
    entry = LedgerEntry(purpose=purpose, note=note, recorded_at=recorded_at)
    added = 0
    for answer in answers:
        if answer_id(answer) not in ledger.answers:
            ledger.answers[answer_id(answer)] = entry
            added += 1
    return added


def parse_freeze(text: str) -> JudgeFreeze:
    try:
        return JudgeFreeze.model_validate_json(text)
    except ValidationError as exc:
        raise EvaluationFileError("It isn't a valid judge freeze.") from exc


def parse_ledger(text: str) -> Ledger:
    try:
        return Ledger.model_validate_json(text)
    except ValidationError as exc:
        raise EvaluationFileError("It isn't a valid ledger.") from exc


def agreement_between(
    first: Mapping[str, MarkerRow], second: Mapping[str, MarkerRow]
) -> MarkerAgreement:
    """Compare two markers on the rows both marked. A blank isn't a disagreement."""
    both = [row for row in first if row in second and first[row].mark and second[row].mark]
    pairs = [(first[row].mark, second[row].mark) for row in both]
    agreed = sum(a == b for a, b in pairs)
    marked_first = {row for row, marks in first.items() if marks.mark}
    marked_second = {row for row, marks in second.items() if marks.mark}
    return MarkerAgreement(
        both_marked=len(both),
        agreed=agreed,
        percent=round(agreed / len(both) * 100, 1) if both else None,
        kappa=cohen_kappa(pairs),
        disagreements=[row for row in both if first[row].mark != second[row].mark],
        marked_by_one_only=len(marked_first ^ marked_second),
    )


def render_adjudication_sheet(
    disagreements: Sequence[str],
    sheet: Mapping[str, Mapping[str, str]],
    first: Mapping[str, MarkerRow],
    second: Mapping[str, MarkerRow],
) -> str:
    """Write a sheet for a third person, showing both markers and never the judge."""
    rows = []
    for row in disagreements:
        cells = sheet[row]
        rows.append(
            (
                row,
                cells.get("area", ""),
                cells.get("customer_asked", ""),
                cells.get("earlier_chat", ""),
                cells.get("what_should_happen", ""),
                cells.get("bot_replied", ""),
                cells.get("bot_did", ""),
                first[row].mark,
                first[row].notes,
                second[row].mark,
                second[row].notes,
                "",
                "",
                "",
            )
        )
    return render_cells(ADJUDICATION_COLUMNS, rows)


def final_marks(
    first: Mapping[str, MarkerRow],
    second: Mapping[str, MarkerRow],
    adjudicated: Mapping[str, MarkerRow],
) -> dict[str, MarkerRow]:
    """Settle each row: what both said, else the adjudicator. Otherwise it stays blank."""
    final: dict[str, MarkerRow] = {}
    for row in first.keys() | second.keys():
        one, two = first.get(row), second.get(row)
        if one and two and one.mark and two.mark and one.mark != two.mark:
            final[row] = adjudicated.get(row, MarkerRow(mark=""))
        else:
            final[row] = (one if one and one.mark else two) or MarkerRow(mark="")
    return final


def assess_independence(
    answers: Sequence[LabelledAnswer], marked_rows: Iterable[str], evidence: EvaluationEvidence
) -> Independence:
    """List everything that stops these figures being called independent. Nothing is assumed."""
    by_row = {labelled.row: labelled.answer for labelled in answers}
    marked = [by_row[row] for row in marked_rows]
    problems = [
        *_freeze_problems(marked, evidence.freeze),
        *_ledger_problems(marked, evidence.ledger),
        *_marker_problems(evidence.agreement),
    ]
    if not evidence.no_ai_statement:
        problems.append("Nobody has declared that the marking used no AI.")
    if len(marked) < MIN_RELIABLE_SAMPLE:
        problems.append(f"Fewer than {MIN_RELIABLE_SAMPLE} answers were marked.")
    return Independence(eligible=not problems, problems=problems)


def _freeze_problems(marked: Sequence[JudgedAnswer], freeze: JudgeFreeze | None) -> list[str]:
    if freeze is None:
        return ["No frozen judge was given."]
    problems = []
    if freeze.judge_id != JUDGE_ID:
        problems.append("The judge has changed since it was frozen.")
    if any(answer.judge_id != freeze.judge_id for answer in marked):
        problems.append("Some verdicts didn't come from the frozen judge.")
    if any(answer.judge_model != freeze.model for answer in marked):
        problems.append("Some verdicts didn't come from the frozen model.")
    return problems


def _ledger_problems(marked: Sequence[JudgedAnswer], ledger: Ledger | None) -> list[str]:
    if ledger is None:
        return ["No ledger was given, so these answers can't be shown to be unseen."]
    seen = sum(answer_id(answer) in ledger.answers for answer in marked)
    return [f"{seen} of these answers were seen before."] if seen else []


def _marker_problems(agreement: MarkerAgreement | None) -> list[str]:
    if agreement is None:
        return ["Only one marker's marks were given."]
    if agreement.kappa is None or agreement.kappa < MIN_MARKER_KAPPA:
        return [f"The two markers' agreement is below the {MIN_MARKER_KAPPA} target."]
    return []
