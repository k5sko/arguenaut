"""Server-side idle auto-shutdown.

When the FastAPI server starts on a Lambda box, it spins up a background thread
that watches the timestamp of the last incoming request. If no request has
arrived for `idle_minutes`, it calls the Lambda Cloud API to terminate its own
instance — so the user never has to remember to run `arguenaut-lambda down`.

Disabled if either LAMBDA_CLOUD_API_KEY or ARGUENAUT_LAMBDA_INSTANCE_ID is unset
(e.g. when running the server locally for development).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class IdleShutdownWatcher:
    def __init__(
        self,
        idle_minutes: float,
        instance_id: str,
        api_key: str,
        check_every_secs: float = 30.0,
        on_shutdown: Callable[[], None] | None = None,
    ):
        self.idle_secs = idle_minutes * 60
        self.instance_id = instance_id
        self.api_key = api_key
        self.check_every = check_every_secs
        self.on_shutdown = on_shutdown
        self._last_request = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def note_request(self) -> None:
        with self._lock:
            self._last_request = time.time()

    def _seconds_idle(self) -> float:
        with self._lock:
            return time.time() - self._last_request

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="idle-shutdown", daemon=True)
        self._thread.start()
        logger.info(
            "Idle auto-shutdown armed: instance %s will terminate after %.0f min idle",
            self.instance_id, self.idle_secs / 60,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        from arguenaut.cloud.lambda_api import LambdaCloudClient, LambdaCloudError

        client = LambdaCloudClient(self.api_key)
        while not self._stop.wait(self.check_every):
            idle = self._seconds_idle()
            if idle < self.idle_secs:
                continue
            logger.warning(
                "Idle for %.0fs (threshold %.0fs); terminating instance %s",
                idle, self.idle_secs, self.instance_id,
            )
            try:
                client.terminate_instances([self.instance_id])
            except LambdaCloudError as e:
                logger.error("Self-terminate failed: %s — will retry on next tick", e)
                continue
            if self.on_shutdown:
                try:
                    self.on_shutdown()
                except Exception:
                    logger.exception("on_shutdown hook raised")
            return


def maybe_install(app, settings) -> IdleShutdownWatcher | None:
    """Attach the watcher + a request middleware to a FastAPI app, if env supports it."""
    instance_id = os.environ.get("ARGUENAUT_LAMBDA_INSTANCE_ID")
    api_key = settings.lambda_cloud_api_key
    minutes = settings.lambda_auto_shutdown_minutes
    if not instance_id or not api_key or minutes <= 0:
        logger.info(
            "Auto-shutdown disabled (instance_id=%s, has_api_key=%s, minutes=%s)",
            bool(instance_id), bool(api_key), minutes,
        )
        return None

    watcher = IdleShutdownWatcher(
        idle_minutes=minutes, instance_id=instance_id, api_key=api_key,
    )

    @app.middleware("http")
    async def _bump(request, call_next):
        # Don't count /health pings against idleness.
        if request.url.path != "/health":
            watcher.note_request()
        return await call_next(request)

    watcher.start()
    return watcher
