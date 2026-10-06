GEMINI_MODEL = "gemini-3.8-flash"
DATE_START = "2021-01-01"
DATE_END   = "2026-08-31"

TICKERS = {
    "^BVSP":  "IBOV",
    "ITUB4.SA": "Itau",
    "VALE3.SA": "Vale",
    "PETR4.SA": "Petrobras",
}

B3_SECTOR_INDEXES = (
    "IFNC",
    "IMAT",
    "IBEE"
)

B3_ASSET_SECTOR_INDEX = {
    "ITUB4.SA" : "IFNC",
    "VALE3.SA" : "IMAT",
    "PETR4.SA" : "IBEE"
}

ROLLING_WINDOW_SIZE = 20