"""Tests for io helpers."""
import pandas as pd

from backtestlibrary.io import ordered_trades_export_columns, reorder_trades_dataframe


def test_ordered_trades_export_columns_groups_entry_exit_continuous():
    cols = ["net_pnl", "Continuous_Col_A_Min", "Entry_Col_Z", "Exit_Col_B", "Continuous_Col_A_Max", "ticker"]
    out = ordered_trades_export_columns(cols)
    assert out[:2] == ["net_pnl", "ticker"]
    assert out[2:3] == ["Entry_Col_Z"]
    assert out[3:4] == ["Exit_Col_B"]
    assert out[4:] == ["Continuous_Col_A_Max", "Continuous_Col_A_Min"]


def test_reorder_trades_dataframe_empty():
    assert reorder_trades_dataframe(pd.DataFrame()).empty


def test_reorder_trades_dataframe_preserves_rows():
    df = pd.DataFrame([{"a": 1, "Entry_Col_x": 2, "Exit_Col_y": 3}])
    r = reorder_trades_dataframe(df)
    assert list(r.columns) == ["a", "Entry_Col_x", "Exit_Col_y"]
    assert r.iloc[0]["a"] == 1
