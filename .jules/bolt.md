## 2026-10-24 - Optimize membership checks using set literals
**Learning:** Using list or tuple literals (e.g., `x in ["a", "b", "c"]`) for membership tests forces Python to either iterate linearly or reconstruct the object at runtime. Python's compiler optimizes set literals (e.g., `x in {"a", "b", "c"}`) into a `frozenset` at compile time, reducing lookup complexity to O(1) and preventing unnecessary runtime overhead.
**Action:** Always use set literals for static membership checks in Python to maximize performance.

## 2026-10-24 - Optimize membership checks using set literals
**Learning:** Using list or tuple literals (e.g., `x in ["a", "b", "c"]` or `x in ("a", "b", "c")`) for membership tests forces Python to either iterate linearly or reconstruct the object at runtime. Python's compiler optimizes set literals (e.g., `x in {"a", "b", "c"}`) into a `frozenset` at compile time, reducing lookup complexity to O(1) and preventing unnecessary runtime overhead. Note that iterating over a set does not grant a performance benefit and introduces non-deterministic order.
**Action:** Always use set literals for static membership checks in Python to maximize performance.

## 2026-10-24 - Optimize membership checks using set literals
**Learning:** Using list or tuple literals (e.g., `x in ["a", "b", "c"]` or `x in ("a", "b", "c")`) for membership tests forces Python to either iterate linearly or reconstruct the object at runtime. Python's compiler optimizes set literals (e.g., `x in {"a", "b", "c"}`) into a `frozenset` at compile time, reducing lookup complexity to O(1) and preventing unnecessary runtime overhead. Note that iterating over a set does not grant a performance benefit and introduces non-deterministic order.
**Action:** Always use set literals for static membership checks in Python to maximize performance.
