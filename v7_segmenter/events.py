"""A minimal publish/subscribe bus: the state publishes, the views listen.

Everything runs on the Tk thread; background results are handed over by the
worker before anything is published.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[], None]]] = defaultdict(list)

    def subscribe(self, topic: str, callback: Callable[[], None]) -> None:
        self._subscribers[topic].append(callback)

    def publish(self, topic: str) -> None:
        for callback in list(self._subscribers[topic]):
            try:
                callback()
            except Exception:   # one broken view must not stop the others
                log.exception("subscriber of %r failed", topic)
