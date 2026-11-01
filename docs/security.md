# Security model

What we protect, how, and what isn't covered. No system can be proven unbreakable, so this file records the risks we know about. Keep it current.

## What matters most

1. **API keys and tokens**, because a stolen key spends money.
2. **Policies and bot replies**, which can be confidential.
3. **Approvals in contracts**, which are audit evidence.
4. **The honesty of verdicts**, because a false pass would mislead a customer.

## Threats and defences

| Threat | Defence | Where |
|---|---|---|
| Key leaks into git | `.env` and results are ignored by git | `.gitignore` |
| Key leaks into logs or errors | Secret types, error types only, no values echoed, crash reports without local variables | `config.py`, `llm.py`, `cli.py` |
| Key sent unencrypted | `https` is required unless the address is on this computer | `config.py`, `target.py` |
| Remote caller uses the bot | Bound to this computer by default. Any other address needs a token of 16+ characters, compared in constant time | `security.py`, `config.py` |
| A website uses a local browser to reach the bot | `Host` and `Origin` checks, and JSON only | `security.py` |
| Huge or slow requests | 128 KB bodies, 10-second body limit, 16 KB headers, 1,000 connections | `security.py`, `__main__.py` |
| Someone runs up the model bill | Rate limit, limit on chats at once, 120-second limit per chat | `api.py`, `security.py` |
| Malformed input | Strict validation that refuses unknown fields, with sizes limited | `schemas.py` |
| A bot's reply injects code into a report | Everything escaped, no JavaScript, and a browser rule blocking anything external | `report.py`, `report.html.j2` |
| A malicious YAML file runs code | Safe loading only, with duplicate and merge keys refused | `yaml_files.py` |
| A crafted file freezes the engine | Size limits, and reading written to avoid slow patterns, with timing tests | `documents.py` |
| A hostile bot sends a token elsewhere, or sends endless data | Redirects off, 1 MB responses, a limit on total time | `target.py` |
| Hidden text steers the AI | Comments and front matter dropped. Documents and replies are sent as data | `documents.py`, `extract.py`, `judge.py` |
| The AI invents evidence | Code confirms every quote, and checks money and tool use itself | `quotes.py`, `judge.py` |
| Results overwrite something else | Inputs are never overwritten, existing results need `--force`, and planted links are refused | `commands/common.py` |
| A compromised dependency | Exact versions and hashes locked, vulnerability scan in CI, weekly update checks | `uv.lock`, `ci.yml`, `dependabot.yml` |
| A compromised CI action | Actions pinned to exact commits, with read-only permissions | `ci.yml` |

## What isn't covered

These are real gaps. Close them before the matching step.

**Before any customer's data:**
- No outside security review or penetration test has been done. Everything above was tested by the person who built it.
- Secrets in `.env` are plain text. Any program running under your account can read them, and so can backups. Secrets Manager is planned for hosting.
- Reports and results aren't encrypted at rest. They rely on file permissions and disk encryption.
- No personal data is stripped from policies or replies yet, and text is sent to the model provider.

**Before putting the bot on the internet:**
- It has no HTTPS of its own, so it needs a proxy that provides it.
- Rate limits are per process and shared by all clients. A proxy should limit per client, and the provider's spending limit is the real ceiling.
- It has one shared token, not separate users, and no audit log.

**Before hosting for several customers:**
- A target URL is trusted because its owner wrote it. A hosted version must block internal addresses, or customers could reach our private network.
- Nothing separates one customer's data from another's, as the engine runs for one user on one machine.

**Can't be fully fixed:**
- Prompt injection can be reduced, but no one can fully prevent it. A visible instruction in a policy has real words, so it can pass the quote check. That's why people review contracts, and findings are checked by hand.
- The judge can be wrong, and its accuracy hasn't been independently measured.
- Most of the code depends on outside packages and the model provider, whose own security we rely on.

**GitHub settings that only the owner can change:**
- Two-factor login on GitHub and the Console.
- Requiring pull requests and passing checks on `main`, if the plan allows it on private repositories.
- Secret scanning, if the plan allows it.
