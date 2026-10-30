"""Tools the model can call. Both are simulated: no money moves and no ticket is raised."""

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from demo_bot.llm import ToolSpec


class RefundArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_id: str = Field(pattern=r"^EP-[0-9]{6}$", description="Order reference, e.g. EP-100245.")
    amount_gbp: float = Field(
        strict=True, gt=0, le=10_000, description="Amount to refund, in pounds."
    )
    reason: str = Field(min_length=1, max_length=500, description="Short reason for the refund.")

    @field_validator("amount_gbp")
    @classmethod
    def _check_whole_pence(cls, value: float) -> float:
        if round(value, 2) != value:
            raise ValueError("must be in whole pence.")
        return value


class EscalationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str = Field(min_length=1, max_length=500, description="Why a person should take over.")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool call produced. `output` is JSON text, which is shown to the model."""

    ok: bool
    output: str

    @classmethod
    def error(cls, message: str) -> Self:
        return cls(ok=False, output=json.dumps({"error": message}))


@dataclass(frozen=True, slots=True)
class Tool[ArgumentsT: BaseModel]:
    name: str
    description: str
    arguments: type[ArgumentsT]
    handler: Callable[[ArgumentsT], dict[str, object]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.arguments.model_json_schema())

    def run(self, raw_arguments: str) -> ToolResult:
        """Validate the model's JSON arguments, then call the handler."""
        try:
            arguments = self.arguments.model_validate_json(raw_arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'arguments'}: {error['msg']}"
                for error in exc.errors(include_url=False, include_input=False)
            )
            return ToolResult.error(f"Invalid arguments. {problems}")
        return ToolResult(ok=True, output=json.dumps(self.handler(arguments)))


def _issue_refund(arguments: RefundArguments) -> dict[str, object]:
    return {
        "status": "approved",
        "refund_reference": _new_reference("RF"),
        "amount_gbp": arguments.amount_gbp,
    }


def _escalate_to_human(_arguments: EscalationArguments) -> dict[str, object]:
    return {
        "status": "escalated",
        "ticket_reference": _new_reference("HD"),
        "next_step": "A colleague will email the customer within 1 working day.",
    }


def _new_reference(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


TOOLS: Mapping[str, Tool[Any]] = {
    tool.name: tool
    for tool in (
        Tool(
            name="issue_refund",
            description="Refund money to the customer's original payment method for an order.",
            arguments=RefundArguments,
            handler=_issue_refund,
        ),
        Tool(
            name="escalate_to_human",
            description="Hand the conversation to a human colleague, who will email the customer.",
            arguments=EscalationArguments,
            handler=_escalate_to_human,
        ),
    )
}
