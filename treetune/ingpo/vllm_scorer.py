"""Adapter that turns the SPO vLLM api_base into an `LPScorer`.

vLLM exposes the OpenAI-style `/v1/completions` endpoint.  Setting
`echo=True, logprobs=1, max_tokens=0` makes vLLM return per-token logprobs
for every token in the prompt — exactly what we need to compute
`log pi(y_i | traj(s))`: tokenize the prefix once, then for each y_i send
`prefix + y_i` and sum the logprobs of the tail tokens.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

from treetune.ingpo.lp_scorer import LPScorer


@dataclass
class VLLMLogprobClient:
    api_base: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 120.0
    max_concurrency: int = 64
    _semaphore: Optional[asyncio.Semaphore] = None
    _client: Optional[Any] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None

    def __post_init__(self) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required for VLLMLogprobClient")

    async def _ensure_async_resources(self) -> None:
        """Create asyncio/httpx resources inside the currently running loop.

        Creating asyncio primitives or async HTTP clients before a loop is
        running can later surface as ``AttributeError: 'NoneType' object has no
        attribute 'create_future'`` from ``asyncio`` internals.  The inference
        strategy may also be reused across calls that are wrapped by separate
        ``asyncio.run(...)`` loops, so recreate resources if the active loop
        changes.
        """

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "VLLMLogprobClient.prompt_logprobs must be awaited inside a running asyncio loop"
            ) from exc

        if self._client is not None and self._loop is loop:
            return

        if self._client is not None:
            try:
                await self._client.aclose()
            except RuntimeError:
                # The previous client may belong to an already-closed loop.
                # Dropping it is safer than trying to reuse stale loop-bound
                # resources.
                pass

        self._loop = loop
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._semaphore = None
        self._loop = None

    async def prompt_logprobs(self, prompt: str) -> List[float]:
        """Return per-token logprobs for `prompt`.  Length == #prompt tokens.

        First token's logprob is `None` from vLLM (no preceding context); we
        keep it as `None` so `LPScorer.score_one` can drop it via filter.
        """

        url = f"{self.api_base.rstrip('/')}/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 0,
            "logprobs": 1,
            "echo": True,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        await self._ensure_async_resources()
        assert self._semaphore is not None
        assert self._client is not None
        try:
            async with self._semaphore:
                resp = await self._client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                response_text = exc.response.text[:500]
                raise RuntimeError(
                    "vLLM logprob request failed with HTTP "
                    f"{exc.response.status_code} for url={url!r}, model={self.model!r}: "
                    f"{response_text}"
                ) from exc
            if httpx is not None and isinstance(
                exc,
                (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                    httpx.NetworkError,
                ),
            ):
                raise RuntimeError(
                    f"vLLM logprob connection failed for url={url!r}, "
                    f"model={self.model!r}: {exc!r}"
                ) from exc
            raise
        # vLLM returns choices[0].logprobs.token_logprobs : List[Optional[float]]
        choice = data["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("token_logprobs") or []
        return list(token_logprobs)


def make_lp_scorer(client: VLLMLogprobClient, tokenize_fn) -> LPScorer:
    async def score_fn(prompt: str, **_):
        return await client.prompt_logprobs(prompt)

    return LPScorer(score_fn=score_fn, tokenize_fn=tokenize_fn)
