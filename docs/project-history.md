# Project history

Everything done on this project, in simple words. **Last updated: 1 November 2026.**

This is kept up to date with every change. The technical reasons behind decisions are in `docs/decisions.md`.

## Where things stand

- **What it is:** a tool that tests companies' customer service chatbots against their own policies, especially after someone changes something.
- **What works:** the whole process, from reading a policy to producing a report, using a real AI model on a practice chatbot.
- **Not proven:** how accurate the AI judge is. Nobody independent has measured it, so every finding must be checked by hand.
- **Not done:** no real company has seen or paid for this, and there has been no outside security test.
- **Next:** talk to UK companies with live chatbots, and try to sell a first audit.
- **Waiting on the founder:** merge pull requests 19 and 20.

## What the product does

| Step | What happens |
|---|---|
| `extract` | Reads a policy and lists its facts and rules, each with an exact quote |
| Human review | A person approves each item |
| `generate` | Writes test questions in customers' own words, and adds 31 fixed trick questions |
| `run` | Asks the chatbot every question and saves each reply |
| `judge` | Marks each reply pass, fail or unclear. Code checks money and actions, and AI marks the rest |
| `report` | Shows what failed, how badly, how often, and what's correct |

## Results so far

### Practice chatbot, tested with a real AI model

- All four planted flaws appeared: the out-of-date £500 FAQ, leaked internal notes, a refund from a fake approval email, and legal threats not passed to a person.
- One small edit to the bot's prompt caused most of the leaks and escalation failures.
- A failure nobody planted: the bot often says it has no information about basic facts that are in its policies.

### The AI judge

| Measure, on the same 316 answers | First version | After fixes |
|---|---|---|
| Wrong fails found by hand | 14 | 0 |
| Wrong passes found by hand | 5 | 0 |
| Replies saying £500 that passed | 11 | 0 |
| Settled by code, with no AI | 0 | 69 |

These numbers flatter the fixes, because they were tuned on answers already read. In a check of 150 answers marked by Claude, the judge agreed on 142, but Claude built the judge, so that isn't independent. The judge is still too strict about some harmless details, and it wrongly passed at least one answer after the fixes.

## Change log

Each row is one pull request, the reviewed unit of change. "Merged" means it's in the main code.

### 30 October 2026: the base

| # | What changed, in simple words | Status |
|---|---|---|
| Start | Installed the tools, ran the checks, saved the first version in git, and created a private GitHub repository | Done |
| 1 | Attacked the practice chatbot and fixed crashes and security gaps. Made its planted flaws appear reliably. Tests went from 94 to 292 | Merged |

### 30 October 2026: building the engine

| # | What changed, in simple words | Status |
|---|---|---|
| 2 | Started the engine, with the file format for approved facts and rules | Merged |
| 3 | The engine can read Markdown policies and split them into sections, ignoring hidden text | Merged |
| 4 | Added the quote check, so nothing gets in unless it's in the policy word for word. "£50" can't match "£500" | Merged |
| 5 | Connected the AI model to pull out facts and rules. The AI can't approve its own work | Merged |
| 6 | Added the `extract` command, which refuses to overwrite approvals | Merged |
| 7 | Saved the first real results as examples. The first real run found 25 items, invented none and missed 1 | Merged |
| 8 | Added the test question format and the first 12 fixed trick questions | Merged |
| 9 | The engine writes questions from approved items, and refuses to run if the policy has changed | Merged |
| 10 | The engine can connect to any chatbot through a small settings file | Merged |
| 11 | The engine asks the questions within limits and saves each answer as it arrives | Merged |
| 12 | Added the `generate` and `run` commands | Merged |
| 13 | Added the AI judge, with code checking its evidence | Merged |
| 14 | Added the HTML and JSON report | Merged |
| 15 | Brought 11 to 14 into the main code, as they had merged into the wrong place | Merged |

### 30 October 2026: first real test, and fixing the judge

| # | What changed, in simple words | Status |
|---|---|---|
| 16 | The founder approved 24 items, rejected 2 and added 1. Trick questions went from 12 to 31. Removed `CLAUDE.md` from the repository | Merged |
| Test | Ran the first full test with a real AI model, on 316 answers, and wrote the first report | Done |
| 17 | The judge now reads whole policies, and code checks money and actions. Added a repeatable check using 42 known answers | Merged |
| 18 | Recorded the before-and-after results, and fixed a mistake in test data | Merged |

### 1 November 2026: architecture and security

| # | What changed, in simple words | Status |
|---|---|---|
| 19 | Made search, checks and judging about 7 to 65 times faster on the code's own work, with results unchanged. Added limits on request rate, chats at once, time and header size, plus security headers. Wrote up the architecture and the security model, including remaining gaps | Waiting to merge |
| 20 | Added this project history and its PDF | Waiting to merge |

## Decisions the founder has made

- Build the base on a practice chatbot with deliberate flaws.
- Use the suggested contract review: approve 24 items, reject 2 and add 1.
- Fix the judge before anything else, and keep its verdicts away from customers until it's better.
- Skip hand-marking for now and move on, so the judge's accuracy remains unmeasured.
- Remove `CLAUDE.md` from the repository.
- Leave "Generated with Claude Code" out of pull requests.

## Mistakes and how they were handled

- Stacked pull requests merged into the wrong place. Pull request 15 brought them back, and stacking stopped.
- A first header limit didn't work. An attack test caught it, and it's now enforced in the app.
- Removing `CLAUDE.md` also deleted the founder's local copy, after being told it would stay. It was restored from history.
- A bug in the evidence check turned real failures into "unclear". It was fixed in pull request 17.

## Keeping this updated

Every change adds a row to the change log, and updates "Where things stand" and "Last updated". To rebuild the PDF, run this from the repo root:

`uv run --no-project --with reportlab python tools/build_history_pdf.py`
