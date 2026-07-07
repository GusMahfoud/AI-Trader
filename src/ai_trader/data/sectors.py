"""Static sector map for the fixed 50-name universe.

Frozen alongside the universe list in config.yaml (GICS-style buckets, coarse on
purpose). Used for sector-relative feature ranks; unknown tickers fall into
"other" so a universe edit can never crash feature building.
"""

from __future__ import annotations

from typing import Dict

SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "CSCO": "tech", "INTC": "tech",
    "ORCL": "tech", "IBM": "tech", "TXN": "tech", "QCOM": "tech", "ADP": "tech",
    # Financials
    "JPM": "financials", "BAC": "financials", "WFC": "financials", "GS": "financials",
    "MS": "financials", "C": "financials", "USB": "financials", "AXP": "financials",
    # Health care
    "JNJ": "health", "PFE": "health", "MRK": "health", "ABT": "health",
    "BMY": "health", "LLY": "health", "AMGN": "health", "UNH": "health",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    # Consumer staples
    "PG": "staples", "KO": "staples", "PEP": "staples", "WMT": "staples",
    "COST": "staples", "MO": "staples", "CL": "staples",
    # Consumer discretionary
    "HD": "discretionary", "MCD": "discretionary", "NKE": "discretionary",
    "DIS": "discretionary", "AMZN": "discretionary",
    # Telecom
    "T": "telecom", "VZ": "telecom",
    # Industrials
    "GE": "industrials", "CAT": "industrials", "MMM": "industrials",
    "BA": "industrials", "HON": "industrials", "UPS": "industrials",
    # --- Wide-universe additions (rank_lambdarank_wide experiment) ---
    "ADBE": "tech", "AMD": "tech", "NVDA": "tech", "MU": "tech", "HPQ": "tech",
    "AMAT": "tech", "ADI": "tech", "INTU": "tech", "EBAY": "tech", "EA": "tech",
    "BLK": "financials", "SCHW": "financials", "BK": "financials", "PNC": "financials",
    "COF": "financials", "MET": "financials", "PRU": "financials", "AIG": "financials",
    "ALL": "financials", "TRV": "financials",
    "TMO": "health", "DHR": "health", "MDT": "health", "SYK": "health",
    "BSX": "health", "GILD": "health", "CI": "health", "CVS": "health",
    "HUM": "health", "BDX": "health",
    "OXY": "energy", "EOG": "energy", "HAL": "energy", "DVN": "energy",
    "FCX": "materials", "NEM": "materials", "NUE": "materials", "APD": "materials",
    "GIS": "staples", "K": "staples", "SYY": "staples", "KR": "staples",
    "TJX": "discretionary", "ROST": "discretionary", "YUM": "discretionary",
    "SBUX": "discretionary", "TGT": "discretionary", "LOW": "discretionary",
    "SO": "utilities", "DUK": "utilities", "D": "utilities", "EXC": "utilities",
    "AEP": "utilities", "NEE": "utilities", "ED": "utilities",
    "LMT": "industrials", "NOC": "industrials", "GD": "industrials",
    "DE": "industrials", "EMR": "industrials", "CSX": "industrials",
    "UNP": "industrials", "FDX": "industrials",
}


def sector_of(ticker: str) -> str:
    """Sector bucket for a ticker; 'other' when unmapped."""
    return SECTOR_MAP.get(ticker, "other")
