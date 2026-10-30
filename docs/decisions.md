# Decisions

Newest last. Each entry: what we decided and why. Change a decision by adding a new entry, not by editing an old one.

## 30 October 2026

1. **Business shape.** Python-only stack to start; open-core model (engine Apache-2.0, hosted product paid); AWS London region; Phase 2 only after paying customers; company based in Manchester.
2. **Product name to be decided.** Code, packages and the demo company use neutral or fictional names (`demo-bot`, Example Parcels, Acme Couriers, Contoso Freight), so renaming later touches nothing important.
3. **One repo as a uv workspace.** Each package has its own dependencies and tests: the open-source engine must not pull in the demo bot's web server, and the demo bot never ships with the engine. Shared tool settings live in the root `pyproject.toml`.
4. **Model access through the official `openai` SDK, not LiteLLM.** Almost every provider (OpenAI, Claude, Gemini, Ollama) offers an OpenAI-compatible endpoint, so one well-maintained SDK plus a configurable base URL covers them with far fewer dependencies. LiteLLM's PyPI releases 1.82.7 and 1.82.8 were replaced with a credential stealer in March 2026. Revisit only if we need a provider without an OpenAI-compatible endpoint.
5. **The demo bot's flaws live in content, never code.** Real bots mostly fail after document or prompt edits, which is exactly what we sell change control for. The code stays correct and secure, so a failure we show is always a content failure.
6. **Demo bot API is stateless and observable.** Clients send the whole conversation; responses include the knowledge base sections used and the tool calls made. No session store, nothing shared between users, and tests can check actions, not just wording.
7. **Access control.** Loopback only by default. Any other host requires a bearer token of at least 16 characters, enforced at startup and on every request.
8. **In-house BM25 search for the demo bot.** A few dozen lines, no dependency, deterministic, so tests can pin the retrieval behaviour the planted flaws rely on. The engine stays retrieval-agnostic.
9. **`platform/` is renamed `hosted/`.** A top-level package called `platform` would shadow Python's standard-library `platform` module.
10. **No licence file yet.** Decide before the first public release; the plan is Apache-2.0 for the engine only.
