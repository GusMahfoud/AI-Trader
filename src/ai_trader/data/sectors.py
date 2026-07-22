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
    # --- Canadian universe (TSX, .TO suffix — see experiments/rank_canada.yaml) ---
    "RY.TO": "financials", "TD.TO": "financials", "BNS.TO": "financials",
    "BMO.TO": "financials", "CM.TO": "financials", "NA.TO": "financials",
    "MFC.TO": "financials", "SLF.TO": "financials", "GWO.TO": "financials",
    "POW.TO": "financials", "IFC.TO": "financials",
    "ENB.TO": "energy", "TRP.TO": "energy", "SU.TO": "energy", "CNQ.TO": "energy",
    "CVE.TO": "energy", "IMO.TO": "energy", "PPL.TO": "energy", "ARX.TO": "energy",
    "KEY.TO": "energy",
    "ABX.TO": "materials", "AEM.TO": "materials", "FNV.TO": "materials",
    "WPM.TO": "materials", "CCO.TO": "materials", "LUN.TO": "materials",
    "FM.TO": "materials", "K.TO": "materials",
    "CNR.TO": "industrials", "CP.TO": "industrials", "TFII.TO": "industrials",
    "CAE.TO": "industrials", "BBD-B.TO": "industrials",
    "BCE.TO": "telecom", "T.TO": "telecom", "RCI-B.TO": "telecom",
    "FTS.TO": "utilities", "EMA.TO": "utilities", "CU.TO": "utilities",
    "TA.TO": "utilities",
    "L.TO": "staples", "MRU.TO": "staples", "ATD.TO": "staples",
    "SAP.TO": "staples", "WN.TO": "staples", "EMP-A.TO": "staples",
    "MG.TO": "discretionary", "GIL.TO": "discretionary", "DOL.TO": "discretionary",
    "CTC-A.TO": "discretionary",
    "CSU.TO": "tech", "OTEX.TO": "tech", "GIB-A.TO": "tech", "BB.TO": "tech",
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
    "RY.TO": "Royal Bank of Canada", "TD.TO": "TD Bank", "BNS.TO": "Scotiabank",
    "BMO.TO": "Bank of Montreal", "CM.TO": "CIBC", "NA.TO": "National Bank of Canada",
    "MFC.TO": "Manulife", "SLF.TO": "Sun Life", "GWO.TO": "Great-West Lifeco",
    "POW.TO": "Power Corporation", "IFC.TO": "Intact Financial",
    "ENB.TO": "Enbridge", "TRP.TO": "TC Energy", "SU.TO": "Suncor Energy",
    "CNQ.TO": "Canadian Natural Resources", "CVE.TO": "Cenovus Energy",
    "IMO.TO": "Imperial Oil", "PPL.TO": "Pembina Pipeline", "ARX.TO": "ARC Resources",
    "KEY.TO": "Keyera", "ABX.TO": "Barrick Gold", "AEM.TO": "Agnico Eagle Mines",
    "FNV.TO": "Franco-Nevada", "WPM.TO": "Wheaton Precious Metals", "CCO.TO": "Cameco",
    "LUN.TO": "Lundin Mining", "FM.TO": "First Quantum Minerals", "K.TO": "Kinross Gold",
    "CNR.TO": "Canadian National Railway", "CP.TO": "Canadian Pacific Kansas City",
    "TFII.TO": "TFI International", "CAE.TO": "CAE", "BBD-B.TO": "Bombardier (B)",
    "BCE.TO": "BCE (Bell Canada)", "T.TO": "Telus", "RCI-B.TO": "Rogers Communications (B)",
    "FTS.TO": "Fortis", "EMA.TO": "Emera", "CU.TO": "Canadian Utilities",
    "TA.TO": "TransAlta", "L.TO": "Loblaw", "MRU.TO": "Metro",
    "ATD.TO": "Alimentation Couche-Tard", "SAP.TO": "Saputo", "WN.TO": "George Weston",
    "EMP-A.TO": "Empire (Sobeys, A)", "MG.TO": "Magna International", "GIL.TO": "Gildan",
    "DOL.TO": "Dollarama", "CTC-A.TO": "Canadian Tire (A)",
    "CSU.TO": "Constellation Software", "OTEX.TO": "OpenText",
    "GIB-A.TO": "CGI (A)", "BB.TO": "BlackBerry",
    "XIU.TO": "iShares S&P/TSX 60 ETF",
}


def name_of(ticker: str) -> str:
    """Company name for a ticker; the ticker itself when unmapped."""
    return TICKER_NAMES.get(ticker, ticker)
