"""
OOP layer for the project.

BaseRecord is an ABSTRACT base class (abstraction) -> Product and
DecisionRecord INHERIT from it and each give their own version of
summary() (polymorphism -- same method name, different behaviour per class).

We also overload a couple of dunder methods (operator overloading):
    __repr__  -> nice printable form
    __len__   -> len(product) returns number of features (so "len()" means
                 something sensible for a Product, duck-typed like a list)
    __call__  -> calling a DecisionRecord instance re-prints itself; shows
                 that instances of our own classes can be "callable" too,
                 same as functions
    __eq__    -> two DecisionRecords are equal if product+decision match
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List


class BaseRecord(ABC):
    """Abstract base class. You cannot do BaseRecord() directly -- it only
    exists so that Product and DecisionRecord are forced to implement
    summary(). This is 'abstraction': we describe *what* every record must
    be able to do, not *how* each one does it."""

    @abstractmethod
    def summary(self) -> str:
        raise NotImplementedError


@dataclass
class Product(BaseRecord):
    """The structured, LLM-extracted view of whatever the user typed in
    plain English. This is what the agent's tools actually check the
    rules against -- not the raw text."""

    name: str
    price: float
    category: str
    features: List[str] = field(default_factory=list)

    def summary(self) -> str:
        # Polymorphism: this is a *different* summary() implementation
        # than DecisionRecord.summary() below, even though the method
        # name and the abstract contract are identical.
        return f"{self.name} ({self.category}) - Rs.{self.price:.0f}"

    def __len__(self) -> int:
        # Operator overloading: len(product) now means "how many
        # features does this product have", the same way len(list)
        # means "how many items". This is duck typing in action --
        # Product now *behaves* like a sized container.
        return len(self.features)

    def __repr__(self) -> str:
        return f"Product(name={self.name!r}, price={self.price}, features={self.features})"


@dataclass
class DecisionRecord(BaseRecord):
    """One row of the decision history log (what gets written to
    data/decision_history.json by the log_decision tool)."""

    product_name: str
    price: float
    decision: str        # "Buy" / "Don't Buy" / "Consider"
    reason: str
    confidence: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def summary(self) -> str:
        return f"[{self.timestamp}] {self.product_name}: {self.decision} ({self.confidence})"

    def __call__(self) -> str:
        # Operator overloading again: record() acts like calling a
        # function, but it's actually an *instance* of our class.
        # Handy for the debug expander in the Streamlit UI.
        return self.summary()

    def __eq__(self, other) -> bool:
        if not isinstance(other, DecisionRecord):
            return NotImplemented
        return self.product_name == other.product_name and self.decision == other.decision

    def __repr__(self) -> str:
        return f"DecisionRecord({self.summary()!r})"

    def to_dict(self) -> dict:
        return {
            "product_name": self.product_name,
            "price": self.price,
            "decision": self.decision,
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "DecisionRecord":
        return DecisionRecord(
            product_name=d["product_name"],
            price=d["price"],
            decision=d["decision"],
            reason=d["reason"],
            confidence=d["confidence"],
            timestamp=d.get("timestamp", ""),
        )
