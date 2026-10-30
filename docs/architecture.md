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

## Tech stack

Python 3.12+, uv workspace, FastAPI, Pydantic, Typer (engine CLI), httpx2, the official `openai` SDK for any OpenAI-compatible model endpoint, pypdf or pdfplumber (avoid PyMuPDF, AGPL), Jinja2. Quality gates: ruff (including security rules), mypy strict, pytest with coverage, pip-audit, all in GitHub Actions. SQLite in Phase 1; Postgres + pgvector, S3 and a Postgres-backed job queue in Phase 2. Docker and Terraform on AWS London (eu-west-2) for UK data residency. Personal data stripped on ingest (e.g. Presidio). Secrets in AWS Secrets Manager, never in code or chat.

## Build order

1. Repo, tooling, CI (done)
2. Demo bot with planted flaws (done)
3. extract + contract format + quote checking (done; Markdown only so far)
4. generate + run with the web-endpoint adapter
5. judge + HTML report: demo ready for customer calls
6. Weeks 2-4: measure the judge against about 150 hand-marked results; SQLite runs + diff; chat-log import; configurable adapters; README; open-source release

## Working agreement (pilot / autopilot)

- Pilot (founder): owns accounts, keys and spending; sets priorities; talks to customers; runs, reviews and merges every change; approves deploys; understands the architecture well enough to explain it to investors and endorsing bodies.
- Autopilot (Claude): designs and writes code, tests, docs, Terraform and CI in small reviewable steps; explains each piece; flags risks.
- Continuity: this file and `docs/decisions.md` are the source of truth. Each session starts by reading them.
