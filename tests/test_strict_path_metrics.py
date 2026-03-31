import pandas as pd


def _mk_bar(close: float, open_: float, high: float, low: float, vwap: float = 0.0):
    return {"Close": close, "Open": open_, "High": high, "Low": low, "Col_VWAP": vwap, "datetime": pd.Timestamp("2022-01-03 09:30")}


def test_half_target_first_ordering_long():
    from backtestlibrary.trade_enrichment import _apply_strict_path_behavior_metrics

    # Entry at 100, stop 98 (risk 2), target 104, half-target = 102
    tdict = {"initial_stop": 98.0, "take_profit": 104.0}
    entry_price = 100.0
    side = "long"

    # Path: hits half-target first (high>=102), then later hits stop (low<=98)
    slice_df = pd.DataFrame(
        [
            {"Open": 100, "Close": 101, "High": 102.5, "Low": 99, "Col_VWAP": 101, "datetime": pd.Timestamp("2022-01-03 09:31")},
            {"Open": 101, "Close": 99, "High": 101, "Low": 97.5, "Col_VWAP": 100, "datetime": pd.Timestamp("2022-01-03 09:32")},
        ]
    )
    day_df = slice_df.copy()

    _apply_strict_path_behavior_metrics(
        tdict,
        day_df,
        slice_df,
        entry_idx=0,
        exit_idx=1,
        side=side,
        entry_price=entry_price,
    )
    assert float(tdict["Exit_Col_DidTradeTouchHalfTargetFirst"]) == 1.0


def test_half_target_first_ordering_long_stop_first():
    from backtestlibrary.trade_enrichment import _apply_strict_path_behavior_metrics

    tdict = {"initial_stop": 98.0, "take_profit": 104.0}
    entry_price = 100.0
    side = "long"

    # Path: hits stop first, then half-target later.
    slice_df = pd.DataFrame(
        [
            {"Open": 100, "Close": 99, "High": 101, "Low": 97.5, "Col_VWAP": 100, "datetime": pd.Timestamp("2022-01-03 09:31")},
            {"Open": 99, "Close": 101, "High": 102.5, "Low": 99, "Col_VWAP": 101, "datetime": pd.Timestamp("2022-01-03 09:32")},
        ]
    )
    day_df = slice_df.copy()

    _apply_strict_path_behavior_metrics(
        tdict,
        day_df,
        slice_df,
        entry_idx=0,
        exit_idx=1,
        side=side,
        entry_price=entry_price,
    )
    assert float(tdict["Exit_Col_DidTradeTouchHalfTargetFirst"]) == 0.0


def test_strict_windows_always_emit_10_20_30_50():
    from backtestlibrary.trade_enrichment import _apply_strict_path_behavior_metrics

    tdict = {"initial_stop": 98.0, "take_profit": 104.0}
    entry_price = 100.0
    side = "long"

    slice_df = pd.DataFrame(
        [
            {"Open": 100, "Close": 100, "High": 101, "Low": 99, "Col_VWAP": 100, "datetime": pd.Timestamp("2022-01-03 09:30")},
            {"Open": 100, "Close": 99, "High": 100, "Low": 98, "Col_VWAP": 99.5, "datetime": pd.Timestamp("2022-01-03 09:31")},
            {"Open": 99, "Close": 101, "High": 102, "Low": 99, "Col_VWAP": 100.5, "datetime": pd.Timestamp("2022-01-03 09:32")},
        ]
    )
    day_df = slice_df.copy()

    _apply_strict_path_behavior_metrics(
        tdict,
        day_df,
        slice_df,
        entry_idx=2,
        exit_idx=2,
        side=side,
        entry_price=entry_price,
    )

    for n in (10, 20, 30, 50):
        assert f"Entry_Col_NumberOfRedBars_Last{n}" in tdict
        assert f"Exit_Col_NumberOfRedBars_Last{n}" in tdict
        assert f"Entry_Col_PercentBarsClosingInUpperQuartile_Last{n}" in tdict
        assert f"Exit_Col_PercentBarsClosingInUpperQuartile_Last{n}" in tdict

