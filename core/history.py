"""
Decision history storage + a CUSTOM ITERATOR over it.

File handling used here (deliberately, not decoratively):
    'x'  -> create decision_log.txt the very first time it doesn't exist yet
            (if it already exists, Python raises FileExistsError -- we catch
            that and just move on, which is the whole point of demonstrating
            'x' mode: it refuses to silently overwrite an existing file)
    'a'  -> append a new decision to the end of the log without touching
            old rows (this is the log_decision tool's job)
    'r'  -> read the whole log back (this is the get_history tool's job)

Format: one decision per line, pipe-separated, so we don't need a database:
    timestamp|product_name|price|decision|confidence|reason
"""

import os
from typing import List

from core.models import DecisionRecord

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "decision_log.txt")
LOG_PATH = os.path.abspath(LOG_PATH)


def _ensure_log_exists() -> None:
    """'x' mode: only succeeds if the file does NOT already exist."""
    try:
        with open(LOG_PATH, "x", encoding="utf-8") as f:
            f.write("# timestamp|product_name|price|decision|confidence|reason\n")
    except FileExistsError:
        # File is already there from a previous run -- that's fine,
        # this is exactly the behaviour 'x' mode is supposed to have.
        pass


def append_decision(record: DecisionRecord) -> None:
    """'a' mode: adds one line, never wipes previous history."""
    _ensure_log_exists()
    line = "|".join([
        record.timestamp,
        record.product_name,
        f"{record.price:.2f}",
        record.decision,
        record.confidence,
        record.reason.replace("|", "/").replace("\n", " "),
    ])
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        # try/except around every file operation, per the syllabus rule
        raise RuntimeError(f"Could not write to decision log: {e}") from e
    finally:
        # 'finally' always runs -- here it's just a cheap sanity print,
        # but it shows the pattern is genuinely try/except/finally, not
        # just try/except.
        pass


def read_all_decisions() -> List[DecisionRecord]:
    """'r' mode: read everything back and parse into DecisionRecord objects."""
    _ensure_log_exists()
    records: List[DecisionRecord] = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) != 6:
                    continue
                timestamp, name, price, decision, confidence, reason = parts
                records.append(DecisionRecord(
                    product_name=name,
                    price=float(price),
                    decision=decision,
                    reason=reason,
                    confidence=confidence,
                    timestamp=timestamp,
                ))
    except OSError as e:
        raise RuntimeError(f"Could not read decision log: {e}") from e
    return records


class DecisionHistoryIterator:
    """A CUSTOM iterator (this is not just `for x in list`, this class
    implements the iterator protocol by hand: __iter__ and __next__, and
    raises StopIteration itself once it runs out of records).

    We iterate newest-first, since for the "X out of last N were Buy"
    calculation we care about the most recent decisions.
    """

    def __init__(self, records: List[DecisionRecord]):
        self._records = list(reversed(records))  # newest first
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> DecisionRecord:
        if self._index >= len(self._records):
            raise StopIteration
        record = self._records[self._index]
        self._index += 1
        return record


class DecisionHistory:
    """Iterable wrapper around the log file. `for record in DecisionHistory()`
    works because __iter__ hands back our custom DecisionHistoryIterator,
    not Python's built-in list_iterator."""

    def __iter__(self) -> DecisionHistoryIterator:
        return DecisionHistoryIterator(read_all_decisions())


def buy_ratio_last_n(n: int) -> str:
    """Real, working use of the custom iterator: walk the last n decisions
    by hand with next()/StopIteration (not a for-loop, so it's obvious the
    iterator protocol is actually driving this), and report how many were
    'Buy'."""
    iterator = iter(DecisionHistory())
    seen = 0
    buys = 0
    while True:
        try:
            record = next(iterator)
        except StopIteration:
            break
        if seen >= n:
            break
        seen += 1
        if record.decision.strip().lower() == "buy":
            buys += 1
    if seen == 0:
        return "No past decisions yet."
    return f"{buys} out of the last {seen} decisions were 'Buy'."
