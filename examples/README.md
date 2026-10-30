# Examples

## targets/

`demo-bot.yaml` tells the engine how to reach the demo bot. Any bot with a web endpoint can be described the same way, with no code.

## contracts/

Contracts extracted by `audit-engine extract` from the demo bot's two current policies, `compensation_policy.md` and `delivery.md`, using `claude-haiku-5-5` on 30 October 2026. A person reviewed them and approved every remaining item.

The 2023 refunds FAQ and the internal notes are left out on purpose: they are the bot's planted flaws, so treating them as policy would make their wrong answers "correct".

What the review changed:

- **Added `delivery-17`:** `delivery.md` says customers can track parcels on the website or app, and the model missed it.
- **Rejected `delivery-10`:** it said what counts as "late", but the policy only says when the refund applies.
- **Rejected `compensation_policy-4`:** it duplicated `compensation_policy-5`, which has the 14-day value to test.
