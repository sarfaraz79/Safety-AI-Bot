# AI agent change control (working title)

Re-tests customer-facing AI agents whenever their knowledge, prompts, model or tools change. The product name is still to be decided, so the code uses neutral names.

## What's here

| Path | What it is |
|---|---|
| `demo-bot/` | A support chatbot with deliberate flaws, used as the test target ([README](demo-bot/README.md)) |
| `engine/` | The audit engine, in progress ([README](engine/README.md)) |
| `docs/` | `architecture.md` (what we're building) and `decisions.md` (why) |
| `CLAUDE.md` | Instructions Claude Code reads at the start of every session |
| `pyproject.toml` | The workspace, plus shared lint, type-check and test settings |
| `uv.lock` | Exact versions of every dependency |

## Set up (once)

1. Install [uv](https://docs.astral.sh/uv/), which also installs the right Python:
   - macOS or Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. In this folder, run `uv sync`.
3. To run the bot (not needed for tests), copy `.env.example` to `.env` and add a model API key.

## Everyday commands

| Command | What it does |
|---|---|
| `uv run demo-bot` | Starts the bot at http://127.0.0.1:8000 (try it at `/docs`) |
| `uv run pytest --cov` | Runs the tests; coverage must stay at 95% or more |
| `uv run ruff format .` | Formats the code |
| `uv run ruff check .` | Lints, including security rules |
| `uv run mypy` | Strict type check |

CI runs all of these, plus a check for known vulnerabilities in dependencies, on every push and pull request.
