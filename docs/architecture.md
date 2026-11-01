# Architecture and build plan

Last updated 30 October 2026. Keep this file current: it is the source of truth.

## What we're building

Change control for customer-facing AI agents. The product re-tests an agent whenever its knowledge, prompts, model or tools change, using tests generated from the company's own policy documents. It produces evidence mapped to UK consumer law (CMA guidance on AI agents) and the FCA Consumer Duty.

- Phase 1: the audit engine (open source, Apache-2.0). Powers paid audits.
- Phase 2: hosted change control (paid). Built only once paying customers ask for it.

## Repo layout

One repo, organised as a uv workspace: each package has its own folder, dependencies and tests, and shares the lint, type-check and test settings in the root `pyproject.toml`.

```
demo-bot/    test-target chatbot (done)
engine/      audit engine: ingest, extract, generate, adapters, judge, report, store, cli (in progress)
examples/    sample contracts, later a sample report
hosted/      phase 2: api, workers, watchers, impact, gate, web
infra/       Terraform
docs/        this file and decisions.md
```

Folders are added when they get their first real content.

## Phase 1: audit engine (weeks 1-4)

### demo-bot/ (done)

A FastAPI chatbot for Example Parcels, a fictional UK courier. Stateless `POST /chat`: the client sends the whole conversation and gets back the reply, the knowledge base sections used (`sources`) and the tools called (`tool_calls`), so tests can check behaviour, not just wording.

Inside: BM25 search over a Markdown knowledge base, a versioned system prompt (`v1`, `v2`), two simulated tools (`issue_refund`, `escalate_to_human`) and an OpenAI-compatible model client. Its four planted flaws (stale refund FAQ, internal notes in the knowledge base, hidden instructions in a pasted email, the v2 prompt regression) are listed in `demo-bot/README.md`.

### engine/ (Python package + command-line tool), five commands

1. extract: PDFs, web pages, Markdown into contract.yaml (facts and rules). Every item carries a source quote, and the tool rejects any item whose quote isn't found in the document. A human reviews the file.
2. generate: tests per fact and rule (direct, reworded, edge values, multi-turn) plus standard packs (competitor bait, hidden-instruction attempts, escalation, UK consumer rights).
3. run: adapters for a generic web endpoint (config says where the question goes and where the answer comes back), OpenAI-compatible endpoints, and imported chat logs. Async, rate-limited, raw transcripts stored.
4. judge: automatic checks first (amounts, dates, competitor names, forbidden phrases), then an AI judge returning pass, fail or unclear with the offending quote. Unclear goes to a human.
5. report: HTML and JSON grouped by severity, each failure showing the question, answer, correct fact and source page. Plus `diff` between two runs.

Storage: SQLite (runs, test cases, results).

## Phase 2: hosted change control (months 3-6)

- Watchers: webhooks or polling for help-centre and wiki edits (Zendesk, Confluence, Notion, SharePoint), prompt repos (GitHub), model retirements, and changes to connected tool descriptions. Each becomes a change event.
- Impact finder: maps a change to the tests it affects. Exact when the customer's bot reports which document sections it used for each answer; approximate otherwise, by matching the changed section to test questions by meaning. If a fact itself changed (e.g. £50 to £75), a human approves the new value first.
- Gate: Slack and email alerts; a GitHub check for prompt changes; a go/no-go endpoint the customer's content pipeline calls before new documents reach the bot (load into a test copy, run affected tests, promote on pass). For vendor bots (Fin, Zendesk) we can only alert.
- Evidence log: append-only record of changes, test runs and approvals, exportable for auditors and insurers.
- Web app: FastAPI with server-rendered pages (HTMX) to start.

## How it's built

This section describes the system as it exists today. Phase 2 above describes where it's going.

| Pattern | What it means here |
|---|---|
| **Modular monolith, in a monorepo** | One repository and one process per run, with separate packages. The engine and bot share no code |
| **Pipes and filters** | `extract`, `generate`, `run`, `judge` and `report` each read a file and write the next. Any step can be re-run or inspected |
| **Layered** | `commands/` handles the command line, the domain modules hold the logic, and `llm.py` and `target.py` talk to the outside |
| **Ports and adapters** | `LLMClient` is an interface, so changing model provider touches one file. A target file adapts the engine to any bot with no code |
| **Stateless service** | The bot keeps nothing between requests, so more copies can run behind a load balancer |
| **Human in the loop** | Nothing is tested until a person approves it, and the model can't approve anything |
| **Fail closed** | When something's doubtful, the verdict is `unclear` or the run stops. A pass is never assumed |
| **Validate at every boundary** | Files, requests, model replies and bot replies are all treated as untrusted |

### Speed of the main operations

*n* is the size of the input, and limits on size keep each one bounded. Python's dictionaries and sets take constant time on average.

| Operation | Data structure | Cost |
|---|---|---|
| Find a document section by heading | Hash map | O(1) |
| Check a section's quote, after its first use | Cached text | O(quote and section length) |
| Check a test id, or an amount, against a known list | Hash map or set | O(1) |
| Rate limit check | Token bucket | O(1) |
| Next request's start time | Pacer | O(1) |
| Search the knowledge base | Inverted index and bounded heap | O(chunks holding the query's words), not all chunks |
| Read or split a document | Single pass | O(n) |
| Read a transcript or results file | Single pass | O(n) |
| Build a report | Group by test, then sort | O(n log t), for t tests |

Sorting can't be done in O(n) in general, and the report's sort is on a short list of failed tests, so it stays. The real limit on speed is waiting for the model, at around a second per call, and no data structure changes that. Judging can run several answers at once with `--concurrency`.

### Scale and its limits

- **Engine:** tested with 11,000 tests and 100,000 answers using a fake model, taking seconds. Real runs are limited by the model provider's rate limits. It runs on one machine, using files.
- **Bot:** can run as several copies, though each has its own rate limit.
- **Beyond one machine:** needs what Phase 2 plans, namely a database, a job queue and workers. It isn't built, because no customer needs it yet.

Reliability comes from limits on every outside call, answers saved as they arrive, files written all at once, and clear exit codes. It hasn't been run in production. See [security.md](security.md) for the security model and its gaps.

## Tech stack

Python 3.12+, uv workspace, FastAPI, Pydantic, Typer (engine CLI), httpx2, the official `openai` SDK for any OpenAI-compatible model endpoint, pypdf or pdfplumber (avoid PyMuPDF, AGPL), Jinja2. Quality gates: ruff (including security rules), mypy strict, pytest with coverage, pip-audit, all in GitHub Actions. SQLite in Phase 1; Postgres + pgvector, S3 and a Postgres-backed job queue in Phase 2. Docker and Terraform on AWS London (eu-west-2) for UK data residency. Personal data stripped on ingest (e.g. Presidio). Secrets in AWS Secrets Manager, never in code or chat.

## Build order

1. Repo, tooling, CI (done)
2. Demo bot with planted flaws (done)
3. extract + contract format + quote checking (done; Markdown only so far)
4. generate + run with the web-endpoint adapter (built; not yet run against a real model)
5. judge + HTML report (built; not yet run against a real model)
6. Measure the judge against about 150 hand-marked results, then rehearse: demo ready for customer calls (decision 25)
7. Weeks 2-4: SQLite runs + diff; chat-log import; more adapters; README; open-source release

## Working agreement (pilot / autopilot)

- Pilot (founder): owns accounts, keys and spending; sets priorities; talks to customers; runs, reviews and merges every change; approves deploys; understands the architecture well enough to explain it to investors and endorsing bodies.
- Autopilot (Claude): designs and writes code, tests, docs, Terraform and CI in small reviewable steps; explains each piece; flags risks.
- Continuity: this file and `docs/decisions.md` are the source of truth. Each session starts by reading them.
