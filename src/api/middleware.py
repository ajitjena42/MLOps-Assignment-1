"""Middleware components for structured logging, request correlation IDs, and timing."""

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api_access")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming HTTP requests and response performance metrics in structured JSON format."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        # Attach request_id to state
        request.state.request_id = request_id

        response = await call_next(request)

        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = str(process_time_ms)

        # Do not log health check probes excessively to avoid log bloating
        if request.url.path not in ("/health", "/metrics"):
            log_payload = {
                "timestamp": time.time(),
                "request_id": request_id,
                "client_ip": request.client.host if request.client else "unknown",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": process_time_ms
            }
            logger.info(json.dumps(log_payload))

        return response
