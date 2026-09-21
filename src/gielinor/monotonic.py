"""A counter that only ever goes up, and never hands out the same value twice.

This looks too small to deserve a module. It exists because two different parts
of the system depend on the same property for safety reasons, and both of them
are easy to get subtly wrong in a way no ordinary test notices.

`oid` identifies a world object the planner was shown. `seq` identifies a
command the actuator was told to run. In both cases a *reused* identifier is
worse than a missing one: a missing identifier fails a validity check and the
action is dropped, while a reused identifier passes every check and points at
the wrong thing. That is the false positive the whole staleness protocol is
built to prevent, so the counter that prevents it is shared, not duplicated.

Values start at 1. Zero is reserved as "no identity" so that a zero-initialised
struct or a missing JSON field can never be mistaken for a valid reference.
"""

from __future__ import annotations

import itertools
import threading

#: Reserved. Never allocated. Means "no object" / "no command".
NO_ID = 0


class MonotonicCounter:
    """Thread-safe source of strictly increasing, never-recycled integers.

    >>> c = MonotonicCounter()
    >>> c.next(), c.next(), c.next()
    (1, 2, 3)
    >>> c.issued
    3

    There is deliberately no ``free()``, ``release()`` or ``reset()``. Anything
    that would let a value come back is the bug this class exists to make
    impossible to write.
    """

    __slots__ = ("_counter", "_issued", "_lock")

    def __init__(self, start: int = 1) -> None:
        if start <= NO_ID:
            raise ValueError(f"start must be > {NO_ID}; {NO_ID} is reserved")
        self._counter = itertools.count(start)
        self._lock = threading.Lock()
        self._issued = 0

    def next(self) -> int:
        with self._lock:
            self._issued += 1
            return next(self._counter)

    @property
    def issued(self) -> int:
        """How many identifiers have been handed out. Also the high-water mark
        relative to ``start``; useful in traces to see churn without logging
        every allocation."""
        return self._issued
