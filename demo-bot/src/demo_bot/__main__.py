"""Command-line entry point: `uv run demo-bot` or `python -m demo_bot`."""

import logging

import uvicorn
from pydantic import ValidationError

from demo_bot.api import create_app
from demo_bot.config import Settings
from demo_bot.security import MAX_HEADER_BYTES

MAX_CONNECTIONS = 1_000


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as exc:
        raise SystemExit(f"Invalid configuration:\n{describe_settings_errors(exc)}") from None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        server_header=False,
        http="h11",  # the HTTP parser that enforces the header limit below
        h11_max_incomplete_event_size=MAX_HEADER_BYTES,
        limit_concurrency=MAX_CONNECTIONS,
    )


def describe_settings_errors(exc: ValidationError) -> str:
    """List each problem by environment variable name, never echoing values (they may be secret)."""
    lines = []
    for error in exc.errors(include_url=False, include_input=False):
        location = f"DEMO_BOT_{str(error['loc'][0]).upper()}" if error["loc"] else "settings"
        lines.append(f"  {location}: {error['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
