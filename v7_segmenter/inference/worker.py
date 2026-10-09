"""Runs slow work off the Tk thread and hands the results back to it.

Jobs run one at a time, in order, on a single daemon thread -- TensorFlow is
never used from two threads at once. Callbacks always run on the Tk thread.
"""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from typing import Any, Callable

log = logging.getLogger(__name__)

Callback = Callable[[Any], None] | None


class BackgroundWorker:
    def __init__(self, root: tk.Misc, poll_ms: int = 50):
        self._root = root
        self._poll_ms = poll_ms
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._pending = 0
        threading.Thread(target=self._run, name="v7-worker", daemon=True).start()
        root.after(poll_ms, self._poll)

    @property
    def pending(self) -> int:
        return self._pending

    def submit(self, function: Callable, *args, on_success: Callback = None,
               on_error: Callback = None) -> None:
        self._pending += 1
        self._jobs.put((function, args, on_success, on_error))

    def _run(self) -> None:
        while True:
            function, args, on_success, on_error = self._jobs.get()
            try:
                result = function(*args)
            except Exception as error:
                log.exception("background job %s failed", getattr(function, "__name__", function))
                self._results.put((on_error, error))
            else:
                self._results.put((on_success, result))

    def _poll(self) -> None:
        try:
            while True:
                callback, value = self._results.get_nowait()
                self._pending -= 1
                if callback is not None:
                    try:
                        callback(value)
                    except Exception:
                        log.exception("result callback failed")
        except queue.Empty:
            pass
        try:
            self._root.after(self._poll_ms, self._poll)
        except tk.TclError:     # the window has been closed
            pass
