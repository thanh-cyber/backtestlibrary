from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ParquetLoaderConfig:
    """Config for loading parquet flat files (e.g. parquet2 folder)."""

    parquet_dir: str
    cache_file: Optional[str] = None
    years: Optional[list[int]] = None
    ticker_col: str = "ticker"
    timestamp_col: str = "window_start"
    price_col: str = "close"
    volume_col: str = "volume"


@dataclass
class LoaderConfig:
    gapper_dir: str
    cache_file: Optional[str] = None
    file_pattern: str = r"(?:.*pmgap.*?)?([0-9]{4})[.](csv|xlsx)$"


def _to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype in (np.float64, np.int64):
        return s.astype(float)
    out = s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(out, errors="coerce")


def _normalize_excel_time_columns(df: pd.DataFrame) -> None:
    new_cols = []
    for c in df.columns:
        if isinstance(c, dt_time):
            if c.second:
                new_cols.append(f"{c.hour}:{c.minute:02d}:{c.second:02d}")
            else:
                new_cols.append(f"{c.hour}:{c.minute:02d}")
        else:
            new_cols.append(c)
    df.columns = new_cols


class GapperDataLoader:
    """Loads and cleans year-partitioned gapper files with optional caching."""

    def __init__(self, config: LoaderConfig):
        self.config = config
        self._pattern = re.compile(config.file_pattern, re.I)

    def list_gapper_files(self) -> list[tuple[int, str]]:
        root = Path(self.config.gapper_dir)
        if not root.is_dir():
            return []
        found: list[tuple[int, str]] = []
        for name in os.listdir(root):
            m = self._pattern.search(name)
            if m:
                year = int(m.group(1))
                found.append((year, str(root / name)))
        return sorted(found)

    def load_cleaned_year_data(self, file_paths: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
        files = file_paths or [path for _, path in self.list_gapper_files()]
        signatures = {p: (os.path.getsize(p), os.path.getmtime(p)) for p in files if os.path.exists(p)}

        if self.config.cache_file:
            cached = self._try_load_cache(signatures)
            if cached is not None:
                return cached

        out: dict[str, pd.DataFrame] = {}
        for path in files:
            m = self._pattern.search(os.path.basename(path))
            if not m:
                continue
            year = m.group(1)
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            _normalize_excel_time_columns(df)
            cleaned = self._clean_df(df)
            if not cleaned.empty:
                out[year] = cleaned

        if self.config.cache_file:
            self._save_cache(signatures, out)
        return out

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Canonical names
        rename_map = {
            "PM High": "PM_High",
            "Premarket High": "PM_High",
            "Previous Close": "Previous_Close",
            "Prev Close": "Previous_Close",
            "Float": "Float_Numeric",
            "Exit Price": "Exit_Price",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "Ticker" in df.columns:
            df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, format="mixed").dt.normalize()

        for col in ("PM_High", "Previous_Close", "Float_Numeric", "Exit_Price"):
            if col in df.columns:
                df[col] = _to_float_series(df[col])

        if "Date" in df.columns:
            df = df.dropna(subset=["Date"])
        if "Ticker" in df.columns:
            df = df[df["Ticker"].notna()]
        return df

    def _try_load_cache(self, signatures: dict[str, tuple[int, float]]) -> Optional[dict[str, pd.DataFrame]]:
        cache_path = Path(self.config.cache_file)
        if not cache_path.exists():
            return None
        try:
            payload = pickle.loads(cache_path.read_bytes())
            if payload.get("signatures") == signatures:
                return payload.get("data", {})
        except Exception:
            return None
        return None

    def _save_cache(self, signatures: dict[str, tuple[int, float]], data: dict[str, pd.DataFrame]) -> None:
        cache_path = Path(self.config.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"signatures": signatures, "data": data}
        cache_path.write_bytes(pickle.dumps(payload))


def _list_parquet_files(root: Path) -> list[tuple[int, str]]:
    """Recursively find .parquet files and infer year from path when present."""
    found: list[tuple[int, str]] = []
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    for path in root.rglob("*.parquet"):
        path_str = str(path)
        m = year_re.search(path_str)
        year = int(m.group(0)) if m else 0
        found.append((year, path_str))
    return sorted(found)


def _pivot_long_to_wide(
    df: pd.DataFrame,
    *,
    ticker_col: str,
    date_col: str,
    time_col: str,
    price_col: str,
    volume_col: str,
) -> pd.DataFrame:
    """Pivot long-format minute bars to wide format for engine compatibility."""
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Date"])
    df = df.copy()
    df["Ticker"] = df[ticker_col].astype(str).str.upper().str.strip()
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["_time"] = df[time_col].astype(str).str.strip()
    df = df.dropna(subset=["Ticker", "Date", "_time", price_col])
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Date"])
    price_pivot = df.pivot_table(
        index=["Ticker", "Date"],
        columns="_time",
        values=price_col,
        aggfunc="last",
    )
    if volume_col in df.columns:
        vol_pivot = df.pivot_table(
            index=["Ticker", "Date"],
            columns="_time",
            values=volume_col,
            aggfunc="sum",
        )
        vol_pivot.columns = [f"Vol {c}" for c in vol_pivot.columns]
        price_pivot = price_pivot.join(vol_pivot)
    price_pivot = price_pivot.reset_index()
    return price_pivot


class ParquetDataLoader:
    """Loads parquet flat files (e.g. parquet2 folder) and produces engine-ready wide format."""

    def __init__(self, config: ParquetLoaderConfig):
        self.config = config

    def list_parquet_files(self) -> list[tuple[int, str]]:
        root = Path(self.config.parquet_dir)
        if not root.is_dir():
            return []
        found = _list_parquet_files(root)
        return found

    def load_cleaned_year_data(self, file_paths: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
        files: list[str]
        if file_paths:
            files = [p for p in file_paths if str(p).lower().endswith(".parquet")]
        else:
            files = [p for _, p in self.list_parquet_files()]
        signatures: dict[str, tuple[int, float]] = {}
        for p in files:
            if os.path.exists(p):
                signatures[p] = (os.path.getsize(p), os.path.getmtime(p))

        if self.config.cache_file and signatures:
            cached = self._try_load_cache(signatures)
            if cached is not None:
                return cached

        tc = self.config.ticker_col
        ts = self.config.timestamp_col
        pc = self.config.price_col
        vc = self.config.volume_col

        parts_by_year: dict[int, list[pd.DataFrame]] = {}
        years_filter = set(self.config.years) if self.config.years else None
        required = {tc, ts, pc}

        for path in files:
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if not required.issubset(df.columns):
                continue
            if vc not in df.columns:
                df[vc] = 1.0

            df[ts] = pd.to_datetime(df[ts], errors="coerce")
            df["_date"] = df[ts].dt.normalize()
            df["_time_str"] = df[ts].dt.strftime("%H:%M")
            df = df.dropna(subset=["_date"])
            if df.empty:
                continue

            df["_year"] = df["_date"].dt.year.astype(int)
            if years_filter is not None:
                df = df[df["_year"].isin(years_filter)]
            if df.empty:
                continue

            for year, grp in df.groupby("_year"):
                parts_by_year.setdefault(int(year), []).append(grp)

        out: dict[str, pd.DataFrame] = {}
        for year, parts in sorted(parts_by_year.items()):
            combined = pd.concat(parts, ignore_index=True)
            combined = combined.drop_duplicates(subset=[tc, "_date", "_time_str"], keep="last")
            wide = _pivot_long_to_wide(
                combined,
                ticker_col=tc,
                date_col="_date",
                time_col="_time_str",
                price_col=pc,
                volume_col=vc,
            )
            if wide.empty:
                continue
            wide["Date"] = pd.to_datetime(wide["Date"], errors="coerce").dt.normalize()
            wide = wide.dropna(subset=["Ticker", "Date"])
            out[str(year)] = wide

        if self.config.cache_file and out:
            self._save_cache(signatures, out)
        return out

    def _try_load_cache(self, signatures: dict[str, tuple[int, float]]) -> Optional[dict[str, pd.DataFrame]]:
        cache_path = Path(self.config.cache_file)
        if not cache_path.exists():
            return None
        try:
            payload = pickle.loads(cache_path.read_bytes())
            if payload.get("signatures") == signatures:
                return payload.get("data", {})
        except Exception:
            return None
        return None

    def _save_cache(self, signatures: dict[str, tuple[int, float]], data: dict[str, pd.DataFrame]) -> None:
        cache_path = Path(self.config.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"signatures": signatures, "data": data}
        cache_path.write_bytes(pickle.dumps(payload))

