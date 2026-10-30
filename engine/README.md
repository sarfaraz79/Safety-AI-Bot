# audit-engine

Tests a customer-facing AI agent against facts and rules taken from a company's own policy documents. The plan is in [docs/architecture.md](../docs/architecture.md).

So far this package holds the contract format (`contract.py`), the Markdown reader (`documents.py`) and the evidence check (`quotes.py`). The `extract` command that produces contracts is still being built.

## Reading policy documents

`documents.py` splits a Markdown document at its headings, at every level, keeping each section's exact words.

- **Left out:** comments and front matter, because readers never see them. A hidden note must not become policy.
- **Kept as text:** tables, lists and code blocks. A `#` inside a code block isn't a heading.
- **Sections:** a heading with no text under it is skipped, and text before the first heading becomes `Untitled section`. Repeated headings are numbered, as in `FAQ (2)`, so each item points to one place.
- **Refused:** files that aren't UTF-8 text, such as a renamed PDF. Also anything over 1 MB, 500 sections or 50,000 characters in a section, because each section costs one model call.
- **Fingerprint:** it covers the original bytes, so any edit to the file changes it, even one that only changes line endings.

## The contract

A contract lists what a policy document says, for a person to review. Every item carries the exact quote it came from.

```yaml
source:
  document: compensation_policy.md
  sha256: 3f1c... # fingerprint of the document, 64 hex characters
items:
  - id: compensation_policy-1
    kind: fact
    statement: Compensation for lost or damaged items is capped per item.
    value: "£50"
    section: Lost or damaged items
    quote: "capped at £50 per item"
    approved: false
  - id: compensation_policy-2
    kind: rule
    statement: Support agents must not approve compensation in chat.
    section: How to claim
    quote: "Support agents cannot approve compensation in chat."
    approved: false
```

- A `fact` has a `value`, such as an amount or a time limit. A `rule` has none.
- `approved` records the reviewer's sign-off. It must be a real `true` or `false`, so a stray `"no"` in quotes can't count as approval.
- Text is kept exactly as written, including spaces, because quotes must match the source word for word.

## Checking the evidence

`quotes.py` checks each item against the document. An item that fails is rejected and listed for the reviewer, never silently dropped.

| Check | Rule |
|---|---|
| Section | The item's `section` must be one of the document's headings |
| Length | A quote needs at least 2 words, so the reviewer can see what it refers to |
| Quote | It must be in that section, word for word, including case and punctuation. Only line breaks and repeated spaces may differ |
| Value | A fact's `value` must be in that item's own quote |

A match can't begin or end inside a longer word or number, so "capped at £50" doesn't match "capped at £500", and "£1" doesn't match "£1,000".

These checks prove the words exist, not that the statement reads them fairly. "refund is available" passes against "No refund is available", and a fact the model leaves out passes everything. Catching those is the job of the human review.
