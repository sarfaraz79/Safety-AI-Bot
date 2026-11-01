"""The web API: GET /health and POST /chat."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from demo_bot.agent import AgentReply, SupportAgent
from demo_bot.config import Settings
from demo_bot.content import load_knowledge_base, load_system_prompt
from demo_bot.llm import AssistantMessage, LLMClient, LLMError, OpenAICompatibleClient, UserMessage
from demo_bot.retrieval import BM25Index
from demo_bot.schemas import ChatRequest, ChatResponse, SourceOut, ToolCallOut
from demo_bot.security import (
    BodySizeLimitMiddleware,
    RateLimiter,
    SecurityHeadersMiddleware,
    require_client_access,
)
from demo_bot.tools import TOOLS

MAX_BODY_BYTES = 128 * 1024

logger = logging.getLogger(__name__)


def create_app(settings: Settings, llm: LLMClient | None = None) -> FastAPI:
    """Build the app. Tests pass `llm` to replace the real model client."""
    model: LLMClient = (
        llm
        if llm is not None
        else OpenAICompatibleClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=str(settings.llm_base_url),
            timeout_seconds=settings.llm_timeout_seconds,
        )
    )
    agent = SupportAgent(
        llm=model,
        index=BM25Index(load_knowledge_base()),
        system_prompt=load_system_prompt(settings.prompt_version),
        tools=TOOLS,
        top_k=settings.retrieval_top_k,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await model.aclose()

    # With a token the bot may be exposed, so it doesn't publish a map of itself.
    exposed = settings.auth_token is not None
    app = FastAPI(
        title="Demo support bot",
        redoc_url=None,
        docs_url=None if exposed else "/docs",
        openapi_url=None if exposed else "/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    app.add_middleware(SecurityHeadersMiddleware)
    rate_limiter = RateLimiter(settings.chats_per_minute)
    chats_running = asyncio.Semaphore(settings.max_concurrent_chats)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {"detail": _describe_problems(exc)}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", dependencies=[Depends(require_client_access(settings.auth_token))])
    async def chat(body: ChatRequest) -> ChatResponse:
        conversation = [
            UserMessage(m.content) if m.role == "user" else AssistantMessage(m.content)
            for m in body.messages
        ]
        # Every chat spends model credits, so refuse excess work quickly rather than queueing it.
        if not rate_limiter.allow():
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Too many requests. Try again shortly.",
                headers={"Retry-After": "5"},
            )
        if chats_running.locked():
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The bot is busy. Try again shortly.",
                headers={"Retry-After": "5"},
            )
        try:
            async with chats_running, asyncio.timeout(settings.chat_timeout_seconds):
                reply = await agent.reply(conversation)
        except TimeoutError as exc:
            logger.warning("Chat exceeded %s seconds.", settings.chat_timeout_seconds)
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT, "The request took too long. Try again later."
            ) from exc
        except LLMError as exc:
            logger.warning("Model request failed: %s", exc)
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "The language model is unavailable. Try again later."
            ) from exc
        logger.info(
            "Chat answered (sources=%d, tool_calls=%d)", len(reply.sources), len(reply.tool_calls)
        )
        return _to_response(reply, settings.prompt_version)

    return app


def _describe_problems(exc: RequestValidationError) -> list[dict[str, object]]:
    """List what's wrong, without FastAPI's echo of the input: it may hold unencodable text."""
    return [
        {
            "type": error["type"],
            "loc": [
                part if isinstance(part, int) else part.encode("utf-8", "replace").decode()
                for part in error["loc"]
            ],
            "msg": error["msg"],
        }
        for error in exc.errors()
    ]


def _to_response(reply: AgentReply, prompt_version: str) -> ChatResponse:
    return ChatResponse(
        reply=reply.text,
        prompt_version=prompt_version,
        sources=[
            SourceOut(id=s.chunk.id, title=s.chunk.title, score=round(s.score, 3))
            for s in reply.sources
        ],
        tool_calls=[
            ToolCallOut(name=c.name, arguments=c.arguments, ok=c.ok, output=c.output)
            for c in reply.tool_calls
        ],
    )
