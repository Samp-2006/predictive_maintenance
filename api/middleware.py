# ============================================================
# api/middleware.py
# ASGI middleware that logs every HTTP request with timing.
# ============================================================

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from utils.logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs method, path, status code, and wall-clock duration for every request.
    Example log line:
        POST /predict 200 OK  143ms
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"{response.status_code} — {elapsed_ms:.1f}ms"
        )
        # Expose timing in response header (useful for frontend dashboards)
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response
