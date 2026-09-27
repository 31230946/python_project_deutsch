"""
A few small, self-contained demos of the "fundamentals" topics that don't
naturally fit inside the agent logic itself, but the instructor may still
poke at in the viva. Wired into the Streamlit debug expander so they're
runnable, not just described.
"""


def my_range(start: int, stop: int = None, step: int = 1):
    """Reimplements range()'s behaviour using a plain while loop + yield,
    to show what's actually happening under the hood of range()/for-loops.
    Supports the same start/stop/step calling convention as real range()."""
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise ValueError("step cannot be 0")

    current = start
    while (step > 0 and current < stop) or (step < 0 and current > stop):
        yield current
        current += step


def custom_for_loop(iterable):
    """Shows what a `for x in iterable:` loop is actually doing internally:
    calling iter() once, then next() repeatedly inside a while loop until
    StopIteration is raised. Returns a list of visited items."""
    visited = []
    it = iter(iterable)
    while True:
        try:
            item = next(it)
        except StopIteration:
            break
        visited.append(item)
    return visited


def introspect(obj) -> list:
    """Thin wrapper around dir(), used by the Streamlit debug expander so
    students can show `dir()` output on the agent / config objects live."""
    return dir(obj)


if __name__ == "__main__":
    # Only runs when this file is executed directly (python core/fundamentals_demo.py),
    # NOT when it's imported by app.py -- classic __name__ == "__main__" guard.
    print("my_range(3, 10, 2):", list(my_range(3, 10, 2)))
    print("custom_for_loop('abc'):", custom_for_loop("abc"))
    print("introspect(42)[:5]:", introspect(42)[:5])
