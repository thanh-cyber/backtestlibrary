import pandas as pd


def test_attach_yfinance_context_populates_all_tickers(monkeypatch):
    import backtestlibrary.librarycolumn_enrichment as le

    # Fake yfinance daily data for sector ETF and benchmarks (only Close needed here).
    idx = pd.to_datetime(["2022-01-03", "2022-01-04"])
    daily_close = pd.DataFrame({"Close": [100.0, 110.0]}, index=idx)

    def fake_daily(symbol: str, start: str, end: str):
        # Return something for all symbols so mapping can occur.
        return daily_close

    def fake_info(tkr: str):
        t = str(tkr).upper()
        if t == "AAA":
            return {"sector": "Technology", "marketCap": 1_000_000, "floatShares": 100_000, "shortPercentOfFloat": 0.10}
        if t == "BBB":
            return {"sector": "Financial", "marketCap": 2_000_000, "floatShares": 200_000, "shortPercentOfFloat": 0.20}
        return {}

    monkeypatch.setattr(le, "_yf_daily", fake_daily)
    monkeypatch.setattr(le, "_yf_info", fake_info)

    long_df = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "BBB", "BBB"],
            "datetime": pd.to_datetime(
                ["2022-01-03 09:30", "2022-01-04 09:30", "2022-01-03 09:30", "2022-01-04 09:30"]
            ),
            "open": [10, 11, 20, 21],
            "high": [10, 11, 20, 21],
            "low": [10, 11, 20, 21],
            "close": [10, 11, 20, 21],
            "volume": [1, 1, 1, 1],
        }
    )

    out = le._attach_yfinance_context(long_df)
    assert "Col_MarketCap" in out.columns
    assert "Col_FloatShares" in out.columns
    assert "Col_ShortInterestPctFloat" in out.columns
    assert "Sector_Close" in out.columns

    a = out[out["Ticker"] == "AAA"].iloc[0]
    b = out[out["Ticker"] == "BBB"].iloc[0]
    assert float(a["Col_MarketCap"]) == 1_000_000.0
    assert float(b["Col_MarketCap"]) == 2_000_000.0
    assert float(a["Col_FloatShares"]) == 100_000.0
    assert float(b["Col_FloatShares"]) == 200_000.0
    assert float(a["Col_ShortInterestPctFloat"]) == 10.0
    assert float(b["Col_ShortInterestPctFloat"]) == 20.0

