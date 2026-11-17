"""
The ONE place that defines which instruments this application may scan.

Scope: NIFTY 50 index and SENSEX index, and their index options (CE and PE) only.
BANKNIFTY, MIDCPNIFTY / MIDCAPNIFTY, FINNIFTY and individual stock F&O are not supported:
every other module takes its instrument list from here, and `validate_indices()` rejects
anything else, including a value typed into config/scanner.ini.

What is fixed here (stable exchange identifiers):
    * the Upstox index instrument key (the underlying; an index is NOT itself tradable)
    * the option / futures "name" used in the Upstox instrument master
    * the F&O segment the option contracts trade on
What is NOT fixed here (read from the Upstox instrument master / option chain every day):
    * expiries, lot sizes, strike intervals, option and futures instrument keys
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

CE, PE = "CE", "PE"
OPTION_TYPES: Tuple[str, str] = (CE, PE)


@dataclass(frozen=True)
class IndexInstrument:
    name: str                    # internal name used everywhere (NIFTY / SENSEX)
    display_name: str
    index_key: str               # Upstox key of the underlying index
    option_name: str             # 'name' column of OPTIDX / FUTIDX rows in the instrument master
    fo_segment: str              # exchange segment of the option contracts
    spot_exchange: str           # NSE_INDEX / BSE_INDEX
    aliases: Tuple[str, ...] = field(default_factory=tuple)   # other spellings seen in instrument masters
    constituents_url: Tuple[str, ...] = field(default_factory=tuple)  # breadth source (empty = proxy)
    breadth_proxy: str = ""      # use this index's breadth when the index has no free constituent list


NIFTY = IndexInstrument(
    name="NIFTY", display_name="NIFTY 50", index_key="NSE_INDEX|Nifty 50", option_name="NIFTY",
    fo_segment="NSE_FO", spot_exchange="NSE_INDEX", aliases=("Nifty 50", "NIFTY 50", "NIFTY"),
    constituents_url=("https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
                      "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"),
)

SENSEX = IndexInstrument(
    name="SENSEX", display_name="SENSEX", index_key="BSE_INDEX|SENSEX", option_name="SENSEX",
    fo_segment="BSE_FO", spot_exchange="BSE_INDEX", aliases=("SENSEX", "BSE SENSEX"),
    # No free NSE list exists for SENSEX: NIFTY 50 breadth is used as its proxy (most SENSEX stocks are
    # NIFTY 50 stocks). The dashboard and alerts label it as a proxy.
    breadth_proxy="NIFTY",
)

INSTRUMENTS: Dict[str, IndexInstrument] = {NIFTY.name: NIFTY, SENSEX.name: SENSEX}
SUPPORTED_INDICES: Tuple[str, ...] = tuple(INSTRUMENTS)
VIX_KEY = "NSE_INDEX|India VIX"


class UnsupportedInstrumentError(ValueError):
    pass


def get_instrument(name: str) -> IndexInstrument:
    key = str(name or "").strip().upper()
    if key not in INSTRUMENTS:
        raise UnsupportedInstrumentError(
            f"'{name}' is not supported. This scanner only covers {', '.join(SUPPORTED_INDICES)} index options (CE/PE).")
    return INSTRUMENTS[key]


def validate_indices(names: Iterable[str]) -> Tuple[str, ...]:
    """Upper-cased, de-duplicated, in the given order. Raises on anything outside NIFTY / SENSEX."""
    out = []
    for raw in names:
        name = get_instrument(raw).name
        if name not in out:
            out.append(name)
    if not out:
        raise UnsupportedInstrumentError("No index selected: choose NIFTY and/or SENSEX.")
    return tuple(out)


def validate_option_type(option_type: str) -> str:
    value = str(option_type or "").strip().upper()
    if value not in OPTION_TYPES:
        raise UnsupportedInstrumentError(f"Option type must be CE or PE, not '{option_type}'.")
    return value


def is_supported_option_row(row: dict) -> bool:
    """True only for an index option (OPTIDX) of NIFTY / SENSEX on its own F&O segment."""
    if str(row.get("instrument_type", "")).upper() != "OPTIDX":
        return False
    inst = INSTRUMENTS.get(str(row.get("name", "")).strip().upper())
    return inst is not None and row.get("exchange") == inst.fo_segment
