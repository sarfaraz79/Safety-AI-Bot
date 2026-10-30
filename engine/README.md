# audit-engine

Tests a customer-facing AI agent against facts and rules taken from a company's own policy documents. The plan is in [docs/architecture.md](../docs/architecture.md).

So far this package holds the contract format, in `contract.py`. The `extract` command that produces contracts is being built next.

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
