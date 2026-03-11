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


def _progress_print_iter(items, desc, total=None):
    """Yield items and print progress to stdout so it's visible in Jupyter/Cursor/VS Code."""
    items = list(items)
    total = total or len(items)
    if total == 0:
        return
    last_pct = -1
    for i, x in enumerate(items):
        pct = (100 * (i + 1)) // total
        if pct != last_pct and (pct % 10 == 0 or pct == 100 or i == 0):
            print(f"{desc}: {pct}% ({i + 1}/{total})", flush=True)
            last_pct = pct
        yield x


@dataclass
class ParquetLoaderConfig:
    """Config for loading parquet flat files (e.g. parquet2 folder)."""

    parquet_dir: str
    cache_dir: Optional[str] = None  # parquet cache dir: {year}.parquet per year + _meta.json
    years: Optional[list[int]] = None
    show_progress: bool = True
    chunk_size: int = 5  # pivot per-file, concat wide in batches; smaller = less peak memory
    stream_to_disk: bool = True  # write batches to temp parquet, avoid holding full year in RAM
    stream_from_cache: bool = True  # when True (default), return dict[str, Path] after building cache to avoid loading full year in RAM (prevents freeze/OOM)
    # Session (hour, minute) ET: only these minute columns are kept in wide cache.
    session_start: tuple[int, int] = (9, 30)  # e.g. (9, 30) = 9:30 AM, (4, 0) = 4:00 AM
    session_end: tuple[int, int] = (16, 0)    # e.g. (16, 0) = 4:00 PM, (9, 30) = 9:30 AM
    ticker_col: str = "ticker"
    timestamp_col: str = "window_start"
    price_col: str = "close"
    volume_col: str = "volume"
    high_col: Optional[str] = "high"  # if set, pivot to "High 9:30" etc. for ORB/strategies that need bar high
    low_col: Optional[str] = "low"  # if set, pivot to "Low 9:30" etc.
    open_col: Optional[str] = "open"  # if set, pivot to "Open 9:30" etc. for full OHLC bar data


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


def _parse_polygon_timestamp(series: pd.Series) -> pd.Series:
    """Parse Polygon-style window_start (nanoseconds UTC) to ET for market-time pivot columns."""
    if pd.api.types.is_integer_dtype(series.dtype):
        ts = pd.to_datetime(series, unit="ns", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(series, errors="coerce")
        if ts.dtype.kind == "M" and getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC", ambiguous="infer")
    if ts.dtype.kind == "M" and getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("America/New_York")
    return ts


def _market_time_str(ts_et: pd.Series) -> pd.Series:
    """Format ET datetime as 'H:MM' or 'HH:MM' to match _standard_minute_columns (e.g. 9:30 not 09:30)."""
    h = ts_et.dt.hour
    m = ts_et.dt.minute
    return h.astype(str) + ":" + m.astype(str).str.zfill(2)


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


def _filter_wide_to_standard_columns(
    wide: pd.DataFrame,
    standard_cols: list[str],
    *,
    keep_high: bool = True,
    keep_low: bool = True,
    keep_open: bool = False,
) -> pd.DataFrame:
    """Keep only standard minute columns; drop extras to prevent memory blowup on concat."""
    if wide.empty:
        return wide
    price_cols = [c for c in standard_cols if c in wide.columns]
    vol_cols = [f"Vol {c}" for c in standard_cols if f"Vol {c}" in wide.columns]
    keep = ["Ticker", "Date"] + price_cols + vol_cols
    if keep_high:
        keep += [f"High {c}" for c in standard_cols if f"High {c}" in wide.columns]
    if keep_low:
        keep += [f"Low {c}" for c in standard_cols if f"Low {c}" in wide.columns]
    if keep_open:
        keep += [f"Open {c}" for c in standard_cols if f"Open {c}" in wide.columns]
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
    extra_value_cols: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Pivot long-format minute bars to wide format for engine compatibility.
    extra_value_cols: optional list of (column_name, prefix) e.g. [('high', 'High '), ('low', 'Low ')].
    """
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
    if extra_value_cols:
        for val_col, prefix in extra_value_cols:
            if val_col not in df.columns:
                continue
            extra = df.dropna(subset=[val_col]).pivot_table(
                index=["Ticker", "Date"],
                columns="_time",
                values=val_col,
                aggfunc="last",
            )
            extra.columns = [prefix + str(c) for c in extra.columns]
            price_pivot = price_pivot.join(extra)
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
        if self.config.cache_dir:
            cached = self._try_load_cache(signatures, years_filter)
            if cached is not None:
                return cached

        tc = self.config.ticker_col
        ts = self.config.timestamp_col
        pc = self.config.price_col
        vc = self.config.volume_col
        required = {tc, ts, pc}
        read_cols = [tc, ts, pc, vc]
        extra_value_cols: list[tuple[str, str]] = []
        if getattr(self.config, "high_col", None):
            extra_value_cols.append((self.config.high_col, "High "))
        if getattr(self.config, "low_col", None):
            extra_value_cols.append((self.config.low_col, "Low "))
        if getattr(self.config, "open_col", None):
            extra_value_cols.append((self.config.open_col, "Open "))
        for col, _ in extra_value_cols:
            if col not in read_cols:
                read_cols.append(col)

        year_to_paths: dict[int, list[str]] = {}
        for y, p in files_with_year:
            if not os.path.exists(p):
                continue
            if years_filter is not None and y > 0 and y not in years_filter:
                continue
            year_to_paths.setdefault(y, []).append(p)

        tqdm_fn = _get_tqdm() if self.config.show_progress else None
        out: dict[str, pd.DataFrame] = {}
        start = getattr(self.config, "session_start", (9, 30))
        end = getattr(self.config, "session_end", (16, 0))
        standard_cols = _standard_minute_columns(start, end)

        chunk_size = max(1, self.config.chunk_size)
        stream_to_disk = getattr(self.config, "stream_to_disk", True)
        import tempfile

        for year in sorted(year_to_paths.keys()):
            if years_filter is not None and year not in years_filter:
                continue
            paths = year_to_paths[year]
            if self.config.show_progress:
                iter_paths = _progress_print_iter(paths, f"Load {year}", len(paths))
            else:
                iter_paths = paths

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

                # Polygon flat files: window_start is nanoseconds UTC -> convert to ET for market-time columns
                ts_et = _parse_polygon_timestamp(df[ts])
                df["_date"] = ts_et.dt.normalize()
                df["_time_str"] = _market_time_str(ts_et)
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
                    extra_value_cols=extra_value_cols if extra_value_cols else None,
                )
                del df
                if wide_one.empty:
                    continue
                wide_one["Date"] = pd.to_datetime(wide_one["Date"], errors="coerce").dt.normalize()
                wide_one = wide_one.dropna(subset=["Ticker", "Date"])
                keep_open = bool(getattr(self.config, "open_col", None))
                wide_one = _filter_wide_to_standard_columns(wide_one, standard_cols, keep_open=keep_open)
                # Avoid duplicate column names (PyArrow rejects them when merging)
                wide_one = wide_one.loc[:, ~wide_one.columns.duplicated()]
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
                # Stream temp parts to cache (or a single temp) without concat in memory
                cache_dir = self._cache_dir() if self.config.cache_dir else None
                merge_path = (cache_dir / f"{year}.parquet") if cache_dir else Path(temp_dir) / "merged.parquet"
                if cache_dir:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except ImportError:
                    pa = pq = None
                if pa is not None and pq is not None:
                    writer = None
                    tqdm_fn = _get_tqdm()
                    if self.config.show_progress:
                        merge_iter = _progress_print_iter(temp_files, f"Merge {year} to cache", len(temp_files))
                    else:
                        merge_iter = temp_files
                    use_pandas_merge = False
                    schema_names: list[str] | None = None
                    for f in merge_iter:
                        tbl = pq.read_table(f)
                        # PyArrow rejects duplicate column names; unify column names if needed
                        if tbl.column_names != list(dict.fromkeys(tbl.column_names)):
                            use_pandas_merge = True
                            del tbl
                            break
                        if writer is None:
                            writer = pq.ParquetWriter(str(merge_path), tbl.schema)
                            schema_names = list(tbl.column_names)
                        else:
                            # Allow parts with identical columns but different order.
                            # Some batches (e.g., early-close days) can produce different pivot column ordering.
                            if schema_names is not None and list(tbl.column_names) != schema_names:
                                if set(tbl.column_names) == set(schema_names):
                                    tbl = tbl.select(schema_names)
                        try:
                            writer.write_table(tbl)
                        except (ValueError, Exception):
                            use_pandas_merge = True
                            if writer is not None:
                                try:
                                    writer.close()
                                except Exception:
                                    pass
                                writer = None
                                if merge_path.exists():
                                    try:
                                        merge_path.unlink()
                                    except OSError:
                                        pass
                            del tbl
                            break
                        del tbl
                        gc.collect()
                    if not use_pandas_merge and writer is not None:
                        writer.close()
                    if use_pandas_merge:
                        # Fallback: concat in pandas (handles schema/duplicate column differences)
                        result = None
                        for tf in temp_files:
                            part = pd.read_parquet(tf)
                            part = part.loc[:, ~part.columns.duplicated()]
                            result = part if result is None else pd.concat([result, part], ignore_index=True, copy=False)
                            del part
                            gc.collect()
                        if result is not None and not result.empty:
                            result = result.loc[:, ~result.columns.duplicated()]
                            result.to_parquet(merge_path, index=False)
                        del result
                        gc.collect()
                    stream_only = getattr(self.config, "stream_from_cache", True) and bool(cache_dir)
                    if stream_only:
                        out[str(year)] = merge_path
                        result = None
                    else:
                        if self.config.show_progress:
                            print(f"Loading {year} from cache into memory...", flush=True)
                        result = pd.read_parquet(merge_path)
                    if not cache_dir and merge_path.exists():
                        try:
                            merge_path.unlink()
                        except OSError:
                            pass
                    if stream_only:
                        for f in temp_files:
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                        try:
                            os.rmdir(temp_dir)
                        except OSError:
                            pass
                        gc.collect()
                        continue
                else:
                    # Fallback: concat in memory (may OOM on large years)
                    result = None
                    for f in temp_files:
                        part = pd.read_parquet(f)
                        result = part if result is None else pd.concat([result, part], ignore_index=True, copy=False)
                        del part
                        gc.collect()
                    if self.config.cache_dir and result is not None and not result.empty:
                        result.to_parquet(merge_path, index=False)
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
                if self.config.cache_dir and result is not None and not result.empty:
                    cache_dir = self._cache_dir()
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    result.to_parquet(cache_dir / f"{year}.parquet", index=False)

            if result is None or result.empty:
                continue
            result = result.drop_duplicates(subset=["Ticker", "Date"], keep="last")
            out[str(year)] = result
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
            # When we have source file list: require cached source files exist and sizes match.
            # Allow signatures to be a superset (e.g. we scan parquet2/, cache was built from parquet2/us_stocks_sip/minute_aggs_v1).
            if signatures:
                meta_sigs = meta.get("signatures", {})
                for k in meta_sigs:
                    if k not in signatures:
                        return None
                    v = signatures[k]
                    meta_val = meta_sigs[k]
                    meta_size = meta_val[0] if isinstance(meta_val, (list, tuple)) else meta_val
                    if meta_size != v[0]:
                        return None
            # Use cache when requested years are present (allow subset of cached years)
            expected_years = meta.get("years", [])
            want_years = (
                {str(y) for y in years_filter}
                if years_filter is not None
                else set(expected_years)
            )
            out: dict[str, pd.DataFrame] = {}
            stream_only = getattr(self.config, "stream_from_cache", True)
            for year in want_years:
                if year not in expected_years:
                    return None
                fp = cache_dir / f"{year}.parquet"
                if not fp.exists():
                    return None
                out[str(year)] = fp if stream_only else pd.read_parquet(fp)
            return out if len(out) == len(want_years) else None
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

