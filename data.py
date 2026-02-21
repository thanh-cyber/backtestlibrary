from __future__ import annotations

import gc
import os
import pickle
import re
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _get_tqdm():
    """Optional tqdm for progress bars."""
    try:
        from tqdm.auto import tqdm
        return tqdm
    except ImportError:
        return None


@dataclass
class ParquetLoaderConfig:
    """Config for loading parquet flat files (e.g. parquet2 folder)."""

    parquet_dir: str
    cache_file: Optional[str] = None
    years: Optional[list[int]] = None
    show_progress: bool = True
    chunk_size: int = 5  # pivot per-file, concat wide in batches; smaller = less peak memory
    stream_to_disk: bool = True  # write batches to temp parquet, avoid holding full year in RAM
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
        year_re = re.compile(r"\b(19|20)\d{2}\b")
        files_with_year: list[tuple[int, str]]
        if file_paths:
            flt = [p for p in file_paths if str(p).lower().endswith(".parquet")]
            files_with_year = []
            for p in flt:
                m = year_re.search(str(p))
                y = int(m.group(0)) if m else 0
                files_with_year.append((y, p))
        else:
            files_with_year = self.list_parquet_files()

        signatures: dict[str, tuple[int, float]] = {}
        for _, p in files_with_year:
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
        years_filter = set(self.config.years) if self.config.years else None
        required = {tc, ts, pc}
        read_cols = [tc, ts, pc, vc]

        year_to_paths: dict[int, list[str]] = {}
        for y, p in files_with_year:
            if not os.path.exists(p):
                continue
            if years_filter is not None and y > 0 and y not in years_filter:
                continue
            year_to_paths.setdefault(y, []).append(p)

        tqdm_fn = _get_tqdm() if self.config.show_progress else None
        out: dict[str, pd.DataFrame] = {}

        chunk_size = max(1, self.config.chunk_size)
        stream_to_disk = getattr(self.config, "stream_to_disk", True)
        import tempfile

        for year in sorted(year_to_paths.keys()):
            if years_filter is not None and year not in years_filter:
                continue
            paths = year_to_paths[year]
            iter_paths = tqdm_fn(paths, desc=f"Load {year}", unit="file") if tqdm_fn else paths

            wide_chunks: list[pd.DataFrame] = []
            temp_files: list[str] = []
            temp_dir: str | None = None

            for path in iter_paths:
                try:
                    df = pd.read_parquet(path, columns=read_cols)
                except Exception:
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
                df = df[df["_year"] == year]
                if df.empty:
                    continue

                wide_one = _pivot_long_to_wide(
                    df,
                    ticker_col=tc,
                    date_col="_date",
                    time_col="_time_str",
                    price_col=pc,
                    volume_col=vc,
                )
                del df
                if wide_one.empty:
                    continue
                wide_one["Date"] = pd.to_datetime(wide_one["Date"], errors="coerce").dt.normalize()
                wide_one = wide_one.dropna(subset=["Ticker", "Date"])
                wide_chunks.append(wide_one)

                if len(wide_chunks) >= chunk_size:
                    batch = pd.concat(wide_chunks, ignore_index=True, copy=False)
                    wide_chunks = []
                    if stream_to_disk:
                        if temp_dir is None:
                            temp_dir = tempfile.mkdtemp(prefix="parquet_loader_")
                        tf = os.path.join(temp_dir, f"batch_{len(temp_files)}.parquet")
                        batch.to_parquet(tf, index=False)
                        temp_files.append(tf)
                        del batch
                    else:
                        if not temp_files:
                            temp_files.append(None)  # placeholder: result in memory
                        if temp_files[0] is None:
                            temp_files[0] = batch
                        else:
                            existing = temp_files[0]
                            temp_files[0] = pd.concat([existing, batch], ignore_index=True, copy=False)
                            del existing
                            del batch
                    gc.collect()

            if wide_chunks:
                batch = pd.concat(wide_chunks, ignore_index=True, copy=False)
                if stream_to_disk and temp_dir:
                    tf = os.path.join(temp_dir, f"batch_{len(temp_files)}.parquet")
                    batch.to_parquet(tf, index=False)
                    temp_files.append(tf)
                elif not stream_to_disk:
                    if not temp_files:
                        temp_files = [batch]
                    else:
                        existing = temp_files[0]
                        temp_files[0] = pd.concat([existing, batch], ignore_index=True, copy=False)
                        del existing
                del batch
                gc.collect()

            if not temp_files:
                continue

            if stream_to_disk and temp_dir:
                result = None
                for f in temp_files:
                    part = pd.read_parquet(f)
                    result = part if result is None else pd.concat([result, part], ignore_index=True, copy=False)
                    del part
                    gc.collect()
                for f in temp_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
            else:
                result = temp_files[0] if isinstance(temp_files[0], pd.DataFrame) else pd.read_parquet(temp_files[0])

            if result.empty:
                continue
            result = result.drop_duplicates(subset=["Ticker", "Date"], keep="last")
            out[str(year)] = result
            gc.collect()

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

