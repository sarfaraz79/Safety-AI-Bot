"""Measures the judge against a person's marks, on a sample the person marks without seeing it."""

import csv
import hashlib
import io
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from audit_engine.results import JudgedAnswer, area_of

MAX_SHEET_ROWS = 100_000
MIN_RELIABLE_SAMPLE = 100
Z_95 = 1.96

SHEET_COLUMNS = (
    "row",
    "area",
    "customer_asked",
    "earlier_chat",
    "what_should_happen",
    "bot_replied",
    "bot_did",
    "your_mark",
    "notes",
)

_CELL_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


class MarksError(ValueError):
    """A marked sheet can't be used."""


class _Mark(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    row: str = Field(min_length=1, max_length=1_000)
    your_mark: Annotated[
        Literal["", "pass", "fail", "unsure"],
        BeforeValidator(lambda mark: mark.strip().lower() if isinstance(mark, str) else mark),
    ]


class Rate(BaseModel):
    """A share with its 95% Wilson interval, in percent. `None` means there was nothing to count."""

    count: int
    out_of: int
    percent: float | None
    low: float | None
    high: float | None


class AreaAccuracy(BaseModel):
    area: str
    marked: int
    missed_failures: Rate
    wrong_fails: Rate


class Accuracy(BaseModel):
    rows: int
    marked: int
    unsure: int
    unmarked: int
    caught_failures: Rate
    missed_failures: Rate
    wrong_fails: Rate
    fails_that_were_real: Rate
    left_for_a_person: Rate
    agreement: Rate
    kappa: float | None
    too_small: bool
    by_area: list[AreaAccuracy]


@dataclass(frozen=True, slots=True)
class LabelledAnswer:
    """One answer, with the id its row carries on the sheet."""

    row: str
    answer: JudgedAnswer


def label_answers(results: Mapping[str, Sequence[JudgedAnswer]]) -> list[LabelledAnswer]:
    """Keep the answers a person can mark, as a bot's silence has nothing to judge."""
    return [
        LabelledAnswer(f"{name}:{answer.case.id}:{answer.attempt}", answer)
        for name, answers in results.items()
        for answer in answers
        if answer.reply is not None
    ]


def choose_sample(
    answers: Sequence[LabelledAnswer], *, size: int, seed: str
) -> list[LabelledAnswer]:
    """Take turns between areas, so rare risks are marked too. Verdicts play no part in choosing."""
    by_area: dict[str, list[LabelledAnswer]] = defaultdict(list)
    for labelled in sorted(answers, key=lambda item: _shuffled_position(item.row, seed)):
        by_area[area_of(labelled.answer)].append(labelled)
    waiting = [by_area[area] for area in sorted(by_area)]
    chosen: list[LabelledAnswer] = []
    while len(chosen) < size and any(waiting):
        for group in waiting:
            if group and len(chosen) < size:
                chosen.append(group.pop(0))
    return sorted(chosen, key=lambda item: _shuffled_position(item.row, seed))


def _shuffled_position(row: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row}".encode()).hexdigest()


def render_sheet(sample: Iterable[LabelledAnswer]) -> str:
    """Write the marking sheet as CSV. It never shows the judge's verdict."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(SHEET_COLUMNS)
    for labelled in sample:
        writer.writerow(_safe_cell(cell) for cell in _sheet_row(labelled))
    return buffer.getvalue()


def _sheet_row(labelled: LabelledAnswer) -> tuple[str, ...]:
    answer = labelled.answer
    case = answer.case
    earlier = "\n".join(f"Customer: {t.customer}\nBot: {t.bot}" for t in answer.earlier)
    wanted = case.expect or " ".join(
        part
        for part in (
            case.statement,
            f"Expected value: {case.expected_value}." if case.expected_value else None,
            f'The policy says: "{case.quote}"',
        )
        if part
    )
    return (
        labelled.row,
        area_of(answer),
        case.question,
        earlier,
        wanted,
        answer.reply or "",
        str(answer.captured) if answer.captured else "",
        "",
        "",
    )


def _safe_cell(text: str) -> str:
    """Stop a spreadsheet running a bot's reply as a formula."""
    return f"'{text}" if text.startswith(_CELL_FORMULA_STARTS) else text


def parse_marks(text: str) -> dict[str, str]:
    """Read a marked sheet, and return each row's mark. Raises MarksError if it can't be used."""
    marks: dict[str, str] = {}
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        for number, line in enumerate(reader, start=1):
            if number > MAX_SHEET_ROWS:
                raise MarksError(f"The sheet has more than {MAX_SHEET_ROWS} rows.")
            mark = _Mark.model_validate(line)
            if mark.row in marks:
                raise MarksError(f"Row {number} repeats an earlier row.")
            marks[mark.row] = mark.your_mark
    except csv.Error as exc:
        raise MarksError("The sheet isn't valid CSV.") from exc
    except ValidationError as exc:
        raise MarksError("Every row needs a row id, and a mark of pass, fail or unsure.") from exc
    return marks


def measure(answers: Sequence[LabelledAnswer], marks: Mapping[str, str]) -> Accuracy:
    """Compare the judge with the person. Raises MarksError for a row it can't match."""
    by_row = {labelled.row: labelled.answer for labelled in answers}
    unknown = marks.keys() - by_row.keys()
    if unknown:
        raise MarksError(f"{len(unknown)} marked rows aren't in these results.")
    marked = [(by_row[row], mark) for row, mark in marks.items() if mark in ("pass", "fail")]
    pairs = [(answer.verdict, mark) for answer, mark in marked]
    decided = [(judge, person) for judge, person in pairs if judge in ("pass", "fail")]
    agreed = sum(judge == person for judge, person in decided)
    judged_fail = [person for judge, person in pairs if judge == "fail"]
    return Accuracy(
        rows=len(marks),
        marked=len(marked),
        unsure=sum(mark == "unsure" for mark in marks.values()),
        unmarked=sum(mark == "" for mark in marks.values()),
        caught_failures=_rate(pairs, judge="fail", person="fail"),
        missed_failures=_rate(pairs, judge="pass", person="fail"),
        wrong_fails=_rate(pairs, judge="fail", person="pass"),
        fails_that_were_real=wilson(judged_fail.count("fail"), len(judged_fail)),
        left_for_a_person=wilson(sum(judge == "unclear" for judge, _ in pairs), len(pairs)),
        agreement=wilson(agreed, len(decided)),
        kappa=_kappa(decided),
        too_small=len(marked) < MIN_RELIABLE_SAMPLE,
        by_area=_by_area(marked),
    )


def _rate(pairs: Sequence[tuple[str, str]], *, judge: str, person: str) -> Rate:
    """Of the answers the person gave this mark, the share the judge called `judge`."""
    theirs = [verdict for verdict, mark in pairs if mark == person]
    return wilson(theirs.count(judge), len(theirs))


def _by_area(marked: Sequence[tuple[JudgedAnswer, str]]) -> list[AreaAccuracy]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for answer, mark in marked:
        groups[area_of(answer)].append((answer.verdict, mark))
    return [
        AreaAccuracy(
            area=area,
            marked=len(pairs),
            missed_failures=_rate(pairs, judge="pass", person="fail"),
            wrong_fails=_rate(pairs, judge="fail", person="pass"),
        )
        for area, pairs in sorted(groups.items())
    ]


def wilson(count: int, out_of: int) -> Rate:
    """Return the share with its Wilson interval, which stays sensible on small samples."""
    if out_of == 0:
        return Rate(count=count, out_of=out_of, percent=None, low=None, high=None)
    share = count / out_of
    z_squared = Z_95**2
    centre = (share + z_squared / (2 * out_of)) / (1 + z_squared / out_of)
    spread = (Z_95 / (1 + z_squared / out_of)) * math.sqrt(
        share * (1 - share) / out_of + z_squared / (4 * out_of**2)
    )
    return Rate(
        count=count,
        out_of=out_of,
        percent=round(share * 100, 1),
        low=round(max(0.0, centre - spread) * 100, 1),
        high=round(min(1.0, centre + spread) * 100, 1),
    )


def _kappa(decided: Sequence[tuple[str, str]]) -> float | None:
    """Return agreement beyond what chance would give. `None` if chance explains everything."""
    if not decided:
        return None
    total = len(decided)
    observed = sum(judge == person for judge, person in decided) / total
    judge_fails = sum(judge == "fail" for judge, _ in decided) / total
    person_fails = sum(person == "fail" for _, person in decided) / total
    chance = judge_fails * person_fails + (1 - judge_fails) * (1 - person_fails)
    if chance == 1:
        return None
    return round((observed - chance) / (1 - chance), 3)
