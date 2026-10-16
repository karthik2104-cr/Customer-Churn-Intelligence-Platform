"""Stop reading a request once it passes the configured upload limit."""

from __future__ import annotations

from fastapi.responses import JSONResponse

from app.core.config import get_settings

_OVERHEAD_BYTES = 1024 * 1024


def body_limit_bytes() -> int:
    """Allow one megabyte of multipart framing above the CSV limit."""
    return get_settings().max_upload_mb * 1024 * 1024 + _OVERHEAD_BYTES


class BodySizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        limit = body_limit_bytes()
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    await _too_large(scope, send)
                    return
            except ValueError:
                pass
        messages = []
        size = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            size += len(message.get("body") or b"")
            if size > limit:
                await _too_large(scope, send)
                return
            if not message.get("more_body", False):
                break
        index = 0

        async def replay():
            nonlocal index
            if index < len(messages):
                current = messages[index]
                index += 1
                return current
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)


async def _too_large(scope, send) -> None:
    response = JSONResponse(
        status_code=413,
        content={
            "code": "file_too_large",
            "message": f"Uploads are limited to {get_settings().max_upload_mb} MB.",
        },
    )

    async def receive():
        return {"type": "http.disconnect"}

    await response(scope, receive, send)
