from __future__ import annotations

import gc
import json
import os
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
    cache_dir: Optional[str] = None  # parquet cache dir: {year}.parquet per year + _meta.json
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
    cache_dir: Optional[str] = None  # parquet cache: {year}.parquet per year + _meta.json
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

        if self.config.cache_dir:
            cached = self._try_load_cache(signatures)
            if cached is not None:
                return cached

        out: dict[str, pd.DataFrame] = {}
        cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
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
                if cache_dir:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cleaned.to_parquet(cache_dir / f"{year}.parquet", index=False)

        if self.config.cache_dir and out:
            self._save_cache_meta(signatures, list(out.keys()))
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
        cache_dir = Path(self.config.cache_dir)
        meta_path = cache_dir / "_meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta_sigs = {k: tuple(v) for k, v in meta.get("signatures", {}).items()}
            if meta_sigs != signatures:
                return None
            expected_years = meta.get("years", [])
            out: dict[str, pd.DataFrame] = {}
            for year in expected_years:
                fp = cache_dir / f"{year}.parquet"
                if fp.exists():
                    out[str(year)] = pd.read_parquet(fp)
            if len(out) != len(expected_years):
                return None
            return out
        except Exception:
            return None

    def _save_cache_meta(self, signatures: dict[str, tuple[int, float]], years: list[str]) -> None:
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {"signatures": {k: list(v) for k, v in signatures.items()}, "years": years}
        with open(cache_dir / "_meta.json", "w") as f:
            json.dump(meta, f)


def _standard_minute_columns(session_start: tuple[int, int] = (9, 30), session_end: tuple[int, int] = (16, 0)) -> list[str]:
    """Return standard minute columns for market session to prevent column explosion across files."""
    cols = []
    start_m = session_start[0] * 60 + session_start[1]
    end_m = session_end[0] * 60 + session_end[1]
    for m in range(start_m, end_m + 1):
        h, mm = divmod(m, 60)
        cols.append(f"{h}:{mm:02d}")
    return cols


def _filter_wide_to_standard_columns(wide: pd.DataFrame, standard_cols: list[str]) -> pd.DataFrame:
    """Keep only standard minute columns; drop extras to prevent memory blowup on concat."""
    if wide.empty:
        return wide
    price_cols = [c for c in standard_cols if c in wide.columns]
    vol_cols = [f"Vol {c}" for c in standard_cols if f"Vol {c}" in wide.columns]
    keep = ["Ticker", "Date"] + price_cols + vol_cols
    return wide[[c for c in keep if c in wide.columns]].copy()


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

        years_filter = set(self.config.years) if self.config.years else None
        if self.config.cache_dir and signatures:
            cached = self._try_load_cache(signatures, years_filter)
            if cached is not None:
                return cached

        tc = self.config.ticker_col
        ts = self.config.timestamp_col
        pc = self.config.price_col
        vc = self.config.volume_col
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
        standard_cols = _standard_minute_columns((9, 30), (16, 0))

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
            accum_df: Optional[pd.DataFrame] = None

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
                wide_one = _filter_wide_to_standard_columns(wide_one, standard_cols)
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
                        accum_df = batch if accum_df is None else pd.concat([accum_df, batch], ignore_index=True, copy=False)
                        del batch
                    gc.collect()

            if wide_chunks:
                batch = pd.concat(wide_chunks, ignore_index=True, copy=False)
                if stream_to_disk:
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp(prefix="parquet_loader_")
                    tf = os.path.join(temp_dir, f"batch_{len(temp_files)}.parquet")
                    batch.to_parquet(tf, index=False)
                    temp_files.append(tf)
                else:
                    accum_df = batch if accum_df is None else pd.concat([accum_df, batch], ignore_index=True, copy=False)
                del batch
                gc.collect()

            if stream_to_disk and not temp_files:
                continue
            if not stream_to_disk and accum_df is None:
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
                result = accum_df

            if result.empty:
                continue
            result = result.drop_duplicates(subset=["Ticker", "Date"], keep="last")
            out[str(year)] = result
            # Write year to disk as we go (streams to parquet; avoids holding all years in memory)
            if self.config.cache_dir:
                cache_dir = self._cache_dir()
                cache_dir.mkdir(parents=True, exist_ok=True)
                result.to_parquet(cache_dir / f"{year}.parquet", index=False)
            gc.collect()

        # Only save cache when all requested years succeeded (avoid partial cache)
        requested = {str(y) for y in year_to_paths.keys()}
        if self.config.cache_dir and out and set(out.keys()) == requested:
            self._save_cache_meta(signatures, list(out.keys()), years_filter)
        return out

    def _cache_dir(self) -> Path:
        """Parquet cache directory (one .parquet per year, _meta.json)."""
        return Path(self.config.cache_dir)

    def _try_load_cache(
        self, signatures: dict[str, tuple[int, float]], years_filter: Optional[set[int]]
    ) -> Optional[dict[str, pd.DataFrame]]:
        cache_dir = self._cache_dir()
        meta_path = cache_dir / "_meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta_sigs = {k: tuple(v) for k, v in meta.get("signatures", {}).items()}
            if meta_sigs != signatures:
                return None
            # Cache invalid if years filter changed (e.g. ran [2022], now [2022,2023])
            want = frozenset(years_filter) if years_filter is not None else None
            got = meta.get("years_filter")
            got_fs = frozenset(got) if got is not None else None
            if want != got_fs:
                return None
            expected_years = meta.get("years", [])
            out: dict[str, pd.DataFrame] = {}
            for year in expected_years:
                fp = cache_dir / f"{year}.parquet"
                if fp.exists():
                    out[str(year)] = pd.read_parquet(fp)
            # Only use cache if all expected files present (avoid partial/corrupt cache)
            if len(out) != len(expected_years):
                return None
            return out
        except Exception:
            return None

    def _save_cache_meta(
        self,
        signatures: dict[str, tuple[int, float]],
        years: list[str],
        years_filter: Optional[set[int]],
    ) -> None:
        """Save cache metadata (parquet files written in loop)."""
        cache_dir = self._cache_dir()
        yf = sorted(years_filter) if years_filter is not None else None
        meta = {"signatures": {k: list(v) for k, v in signatures.items()}, "years": years, "years_filter": yf}
        with open(cache_dir / "_meta.json", "w") as f:
            json.dump(meta, f)

