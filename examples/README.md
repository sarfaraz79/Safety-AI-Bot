# Examples

## contracts/

Contracts extracted by `audit-engine extract` from the demo bot's two current policies, `compensation_policy.md` and `delivery.md`, using `claude-haiku-5-5` on 30 October 2026. They are unedited and **nothing in them is approved**.

The 2023 refunds FAQ and the internal notes are left out on purpose: they are the bot's planted flaws, so treating them as policy would make their wrong answers "correct".

A person would need to fix these before approving:

- **Missing fact:** `delivery.md` says customers can track parcels on the website or app, and no item covers it.
- **Loose reading:** `delivery-10` says what counts as "late", but the policy only says when the refund applies.
- **Duplicates:** `compensation_policy-4` and `-5` both cover the 14-day claim limit, and `delivery-9` and `-10` overlap.
