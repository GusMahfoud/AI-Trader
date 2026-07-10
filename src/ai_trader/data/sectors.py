"""Static universe metadata: sector buckets and company names per ticker.

Frozen alongside the universe list in config.yaml (GICS-style buckets, coarse on
purpose). Sectors feed the sector-relative feature ranks; names make picks
output human-readable. Unknown tickers fall back gracefully ("other" / the
ticker itself) so a universe edit can never crash feature building or picks.
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


TICKER_NAMES: Dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet (Google)", "AMZN": "Amazon",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup", "USB": "U.S. Bancorp",
    "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "MRK": "Merck", "ABT": "Abbott Laboratories",
    "BMY": "Bristol-Myers Squibb", "LLY": "Eli Lilly", "AMGN": "Amgen",
    "UNH": "UnitedHealth Group", "XOM": "Exxon Mobil", "CVX": "Chevron",
    "COP": "ConocoPhillips", "SLB": "SLB (Schlumberger)", "PG": "Procter & Gamble",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "WMT": "Walmart", "COST": "Costco",
    "HD": "Home Depot", "MCD": "McDonald's", "NKE": "Nike", "MO": "Altria",
    "CL": "Colgate-Palmolive", "T": "AT&T", "VZ": "Verizon", "CSCO": "Cisco Systems",
    "INTC": "Intel", "ORCL": "Oracle", "IBM": "IBM", "TXN": "Texas Instruments",
    "QCOM": "Qualcomm", "ADP": "Automatic Data Processing", "GE": "GE Aerospace",
    "CAT": "Caterpillar", "MMM": "3M", "BA": "Boeing", "HON": "Honeywell",
    "UPS": "United Parcel Service", "DIS": "Walt Disney", "AXP": "American Express",
    "ADBE": "Adobe", "AMD": "Advanced Micro Devices", "NVDA": "NVIDIA",
    "MU": "Micron Technology", "HPQ": "HP Inc.", "AMAT": "Applied Materials",
    "ADI": "Analog Devices", "INTU": "Intuit", "EBAY": "eBay", "EA": "Electronic Arts",
    "BLK": "BlackRock", "SCHW": "Charles Schwab", "BK": "BNY Mellon",
    "PNC": "PNC Financial", "COF": "Capital One", "MET": "MetLife",
    "PRU": "Prudential Financial", "AIG": "American International Group",
    "ALL": "Allstate", "TRV": "Travelers", "TMO": "Thermo Fisher Scientific",
    "DHR": "Danaher", "MDT": "Medtronic", "SYK": "Stryker", "BSX": "Boston Scientific",
    "GILD": "Gilead Sciences", "CI": "Cigna", "CVS": "CVS Health", "HUM": "Humana",
    "BDX": "Becton Dickinson", "OXY": "Occidental Petroleum", "EOG": "EOG Resources",
    "HAL": "Halliburton", "DVN": "Devon Energy", "FCX": "Freeport-McMoRan",
    "NEM": "Newmont", "NUE": "Nucor", "APD": "Air Products",
    "GIS": "General Mills", "K": "Kellanova (Kellogg)", "SYY": "Sysco", "KR": "Kroger",
    "TJX": "TJX Companies", "ROST": "Ross Stores", "YUM": "Yum! Brands",
    "SBUX": "Starbucks", "TGT": "Target", "LOW": "Lowe's",
    "SO": "Southern Company", "DUK": "Duke Energy", "D": "Dominion Energy",
    "EXC": "Exelon", "AEP": "American Electric Power", "NEE": "NextEra Energy",
    "ED": "Consolidated Edison", "LMT": "Lockheed Martin", "NOC": "Northrop Grumman",
    "GD": "General Dynamics", "DE": "Deere & Company", "EMR": "Emerson Electric",
    "CSX": "CSX", "UNP": "Union Pacific", "FDX": "FedEx",
}


def name_of(ticker: str) -> str:
    """Company name for a ticker; the ticker itself when unmapped."""
    return TICKER_NAMES.get(ticker, ticker)
