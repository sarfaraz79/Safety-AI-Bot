"""Measures the judge against a person's marks, on a sample the person marks without seeing it."""

import csv
import hashlib
import io
import itertools
import math
from collections import Counter, defaultdict
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
    "your_severity",
    "notes",
)
MAX_DUPLICATE_CHECK_ROWS = 2_000
NEAR_DUPLICATE_SIMILARITY = 0.9
HIGH_SEVERITIES = ("critical", "high")

_CELL_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


class MarksError(ValueError):
    """A marked sheet can't be used."""


def _lowered(value: object) -> object:
    return value.strip().lower() if isinstance(value, str) else value


class MarkerRow(BaseModel):
    """One person's verdict on one answer. `unsure` means the same as `unclear`."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    mark: Annotated[Literal["", "pass", "fail", "unclear"], BeforeValidator(_lowered)]
    severity: Annotated[
        Literal["", "critical", "high", "medium", "low"], BeforeValidator(_lowered)
    ] = ""
    notes: str = Field(default="", max_length=2_000)


class _SheetRow(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    row: str = Field(min_length=1, max_length=1_000)
    your_mark: Annotated[
        Literal["", "pass", "fail", "unclear", "unsure"], BeforeValidator(_lowered)
    ]
    your_severity: str = ""
    notes: str = ""


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
    unclear_marks: int
    unmarked: int
    distinct_tests: int
    near_duplicate_pairs: int | None
    confusion: dict[str, dict[str, int]]
    missed_high_risk: Rate
    missed_high_risk_rows: list[str]
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
    """Keep the answers a person can mark, as silence or a fallback has nothing to judge."""
    return [
        LabelledAnswer(f"{name}:{answer.case.id}:{answer.attempt}", answer)
        for name, answers in results.items()
        for answer in answers
        if answer.reply is not None and answer.verdict != "error"
    ]


def choose_sample(
    answers: Sequence[LabelledAnswer],
    *,
    size: int,
    seed: str,
    one_per_test: bool = False,
    leave_out: frozenset[str] = frozenset(),
) -> list[LabelledAnswer]:
    """Take turns between areas, so rare risks are marked too. Verdicts play no part in choosing."""
    by_area: dict[str, list[LabelledAnswer]] = defaultdict(list)
    tests_taken: set[str] = set()
    for labelled in sorted(answers, key=lambda item: _shuffled_position(item.row, seed)):
        test = labelled.row.rpartition(":")[0]
        if labelled.row in leave_out or (one_per_test and test in tests_taken):
            continue
        tests_taken.add(test)
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
    return render_cells(SHEET_COLUMNS, [_sheet_row(labelled) for labelled in sample])


def render_cells(columns: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Write rows as CSV, so that no cell can run as a spreadsheet formula."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(_safe_cell(cell) for cell in row)
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
        "",
    )


def _safe_cell(text: str) -> str:
    """Stop a spreadsheet running a bot's reply as a formula."""
    return f"'{text}" if text.startswith(_CELL_FORMULA_STARTS) else text


def parse_sheet(text: str) -> tuple[dict[str, MarkerRow], dict[str, dict[str, str]]]:
    """Read a marked sheet's marks and cells. Raises MarksError if it can't be used."""
    marks: dict[str, MarkerRow] = {}
    cells: dict[str, dict[str, str]] = {}
    try:
        # Spreadsheets often save CSV with an invisible byte order mark at the start.
        reader = csv.DictReader(io.StringIO(text.removeprefix("\ufeff"), newline=""), strict=True)
        for number, line in enumerate(reader, start=1):
            if number > MAX_SHEET_ROWS:
                raise MarksError(f"The sheet has more than {MAX_SHEET_ROWS} rows.")
            row = _SheetRow.model_validate(line)
            if row.row in marks:
                raise MarksError(f"Row {number} repeats an earlier row.")
            mark = "unclear" if row.your_mark == "unsure" else row.your_mark
            marks[row.row] = MarkerRow.model_validate(
                {"mark": mark, "severity": row.your_severity, "notes": row.notes}
            )
            cells[row.row] = line
    except csv.Error as exc:
        raise MarksError("The sheet isn't valid CSV.") from exc
    except ValidationError as exc:
        raise MarksError(
            "Every row needs a row id, a mark of pass, fail or unclear, and a known severity."
        ) from exc
    return marks, cells


def parse_marks(text: str) -> dict[str, str]:
    """Read only the marks from a marked sheet."""
    return {row: marks.mark for row, marks in parse_sheet(text)[0].items()}


def measure(
    answers: Sequence[LabelledAnswer],
    marks: Mapping[str, str],
    severities: Mapping[str, str] | None = None,
) -> Accuracy:
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
    high_risk = _high_risk_failures(by_row, marks, severities or {})
    missed = [row for row in high_risk if by_row[row].verdict == "pass"]
    marked_rows = [row for row, mark in marks.items() if mark]
    return Accuracy(
        rows=len(marks),
        marked=len(marked),
        unclear_marks=sum(mark == "unclear" for mark in marks.values()),
        unmarked=sum(mark == "" for mark in marks.values()),
        distinct_tests=len({by_row[row].case.id for row in marked_rows}),
        near_duplicate_pairs=near_duplicate_pairs([by_row[row] for row in marked_rows]),
        confusion=_confusion((by_row[row].verdict, marks[row]) for row in marked_rows),
        missed_high_risk=wilson(len(missed), len(high_risk)),
        missed_high_risk_rows=sorted(missed),
        caught_failures=_rate(pairs, judge="fail", person="fail"),
        missed_failures=_rate(pairs, judge="pass", person="fail"),
        wrong_fails=_rate(pairs, judge="fail", person="pass"),
        fails_that_were_real=wilson(judged_fail.count("fail"), len(judged_fail)),
        left_for_a_person=wilson(sum(judge == "unclear" for judge, _ in pairs), len(pairs)),
        agreement=wilson(agreed, len(decided)),
        kappa=cohen_kappa(decided),
        too_small=len(marked) < MIN_RELIABLE_SAMPLE,
        by_area=_by_area(marked),
    )


def _high_risk_failures(
    by_row: Mapping[str, JudgedAnswer], marks: Mapping[str, str], severities: Mapping[str, str]
) -> list[str]:
    """Rows a person failed that are serious, by their severity or a tool the test checks."""
    return [
        row
        for row, mark in marks.items()
        if mark == "fail"
        and (
            severities.get(row) in HIGH_SEVERITIES
            or by_row[row].case.forbids_action
            or by_row[row].case.requires_action
        )
    ]


def _confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Count each judge verdict against each mark a person gave."""
    table: dict[str, dict[str, int]] = {}
    for judge, person in pairs:
        row = table.setdefault(judge, {})
        row[person] = row.get(person, 0) + 1
    return table


def near_duplicate_pairs(answers: Sequence[JudgedAnswer]) -> int | None:
    """Count pairs with almost the same question and reply. `None` if there are too many."""
    if len(answers) > MAX_DUPLICATE_CHECK_ROWS:
        return None
    words = [frozenset(f"{a.case.question} {a.reply or ''}".casefold().split()) for a in answers]
    return sum(
        len(one & two) >= NEAR_DUPLICATE_SIMILARITY * len(one | two)
        for one, two in itertools.combinations(words, 2)
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


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Return agreement beyond what chance would give. `None` if chance explains everything."""
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(one == two for one, two in pairs) / total
    first, second = Counter(one for one, _ in pairs), Counter(two for _, two in pairs)
    chance = sum(first[label] * second[label] for label in first) / total**2
    if chance == 1:
        return None
    return round((observed - chance) / (1 - chance), 3)
