"""
Broker abstraction for the execution engine. The strategy, signals and paper book never import a broker:
src/execution/order_manager.py talks to an object implementing `Broker`, so another broker can be added by
writing one class (see paper_broker.py and upstox_broker.py) without touching the strategy engine.

Order lifecycle states used by the order manager (broker-specific statuses are mapped onto these):
    PENDING     sent, not yet acknowledged
    OPEN        working at the exchange (may be partially filled)
    FILLED      completely filled
    CANCELLED   cancelled (possibly after a partial fill)
    REJECTED    refused by the broker / exchange (status_message says why)
    UNKNOWN     the broker could not be asked (timeout) - treated as "may be working", never as "not sent"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

PENDING, OPEN, FILLED, CANCELLED, REJECTED, UNKNOWN = "PENDING", "OPEN", "FILLED", "CANCELLED", "REJECTED", "UNKNOWN"
TERMINAL = (FILLED, CANCELLED, REJECTED)


class BrokerError(Exception):
    """A request the broker refused or that could not be completed (network, timeout, validation)."""

    def __init__(self, message: str, retryable: bool = False, maybe_sent: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.maybe_sent = maybe_sent     # True when a placement may have reached the exchange (e.g. timeout)


@dataclass
class OrderRequest:
    instrument_key: str
    side: str                 # BUY / SELL
    quantity: int             # units (lots x lot size)
    order_type: str = "LIMIT"   # LIMIT / MARKET / SL / SL-M
    price: float = 0.0
    trigger_price: float = 0.0
    product: str = "I"        # I = intraday
    validity: str = "DAY"
    tag: str = ""             # idempotency key sent to the broker (<= 20 characters)

    def to_dict(self):
        return asdict(self)


@dataclass
class OrderStatus:
    order_id: str
    status: str               # one of the states above
    quantity: int = 0
    filled_quantity: int = 0
    average_price: float = 0.0
    pending_quantity: int = 0
    message: str = ""
    tag: str = ""
    raw_status: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class BrokerPosition:
    instrument_key: str
    quantity: int             # net quantity (+ long)
    average_price: float = 0.0
    last_price: float = 0.0
    pnl: float = 0.0


@dataclass
class Funds:
    available_margin: float
    used_margin: float = 0.0
    raw: Dict = field(default_factory=dict)


class Broker(ABC):
    name = "broker"
    live = False              # True only for a broker that sends real orders

    @abstractmethod
    def place_order(self, req: OrderRequest) -> List[str]:
        """Returns the broker order id(s). Raises BrokerError (maybe_sent=True when the outcome is unknown)."""

    @abstractmethod
    def modify_order(self, order_id: str, *, quantity: Optional[int] = None, price: Optional[float] = None,
                     trigger_price: Optional[float] = None, order_type: Optional[str] = None) -> None: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def order_status(self, order_id: str = "", tag: str = "") -> Optional[OrderStatus]: ...

    @abstractmethod
    def positions(self) -> List[BrokerPosition]: ...

    @abstractmethod
    def funds(self) -> Funds: ...

    def healthy(self) -> (bool, str):
        """Cheap connectivity check (funds call). Never raises."""
        try:
            self.funds()
            return True, ""
        except Exception as error:   # noqa: BLE001 - report, never crash the caller
            return False, f"{type(error).__name__}: {str(error)[:120]}"
