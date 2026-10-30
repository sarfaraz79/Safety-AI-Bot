# CLAUDE.md

Read `docs/architecture.md` and `docs/decisions.md` at the start of each session; they are the source of truth for what we are building and why.

## The project

The base for a change-control product for customer-facing AI agents (name not chosen yet; keep code names neutral). One repo, organised as a uv workspace. Each package has its own folder, `pyproject.toml` and tests:

- `demo-bot/`: a support chatbot for a fictional courier, Example Parcels, used as the test target.
- `engine/` (next): the audit engine.

## The demo bot is flawed on purpose

Its flaws live in `demo-bot/src/demo_bot/data/` and are listed in `demo-bot/README.md`. Do not fix them, or add guardrails that hide them, unless asked. The code itself must stay correct and secure.

## Commands (from the repo root)

- Install: `uv sync`
- Before saying any change is done, all of these must pass:
  - `uv run ruff format .`
  - `uv run ruff check .`
  - `uv run mypy`
  - `uv run pytest --cov`
- Run the bot: `uv run demo-bot`

## How we write code

- Python 3.12+, fully typed; mypy strict must pass without `# type: ignore`.
- Small modules with one job each. No dead code, commented-out code, speculative options or unused config. If something isn't needed now, leave it out.
- Absolute imports (`from demo_bot.x import y`).
- Validate every external input with Pydantic, including size limits.
- Secrets come from environment variables or `.env` only. Never commit `.env`, and never log secrets or customer message content.
- Every behaviour change comes with tests. Tests never call a real model or the network: use `demo_bot.testing.ScriptedLLM` and `httpx2.MockTransport`.
- Prefer the standard library. Add a dependency only with `uv add`, and say why.
- Record significant decisions in `docs/decisions.md` (date, decision, reason).
- When adding a package, add it to `[tool.uv.workspace]`, and its paths to ruff `src`, mypy `files`, pytest `testpaths` and coverage `source` in the root `pyproject.toml`.

## Working together

The founder is the pilot: owns accounts, keys and spending, reviews every change, and approves merges and deploys. Work in small, reviewable steps, explain what each change does and why, and flag risks plainly.
