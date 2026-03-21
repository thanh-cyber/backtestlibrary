from __future__ import annotations

import gc
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path
from typing import Callable, Optional, Union

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


def _print_progress_bar(
    pct: float,
    current: int,
    total: int,
    desc: str = "Loading",
    width: int = 40,
    *,
    unit: str = "file",
    use_carriage_return: bool = True,
) -> None:
    """Print a single-line progress bar 0–100% that fills as pct increases. Works in terminal and Jupyter."""
    import sys
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(width * pct / 100) if total > 0 else width
    bar = "[" + "#" * filled + " " * (width - filled) + "]"
    tail = f" {pct:.0f}% ({current}/{total} {unit}s)" if total > 0 else " 100%"
    line = f"\r{desc}: {bar}{tail}"
    if use_carriage_return:
        sys.stdout.write(line)
        sys.stdout.flush()
    else:
        print(line, flush=True)


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
    # When True, write cache as three session files (pm / rth / ah) in one pass instead of one full-year file.
    split_sessions: bool = False
    # When loading from a split cache, which session to return (pm / rth / ah). Default rth.
    session_output: str = "rth"
    # Prefix for legacy flat split cache filenames: "<prefix>_<session>_<year>.parquet"
    # Default "normal" (e.g. normal_pm_2026.parquet). Use "nasdaq" for nasdaq_pm_2026.parquet.
    # Backward compatibility: older caches used "vwap_*".
    cache_prefix: str = "normal"
    ticker_col: str = "ticker"
    timestamp_col: str = "window_start"
    price_col: str = "close"
    volume_col: str = "volume"
    high_col: Optional[str] = "high"  # if set, pivot to "High 9:30" etc. for ORB/strategies that need bar high
    low_col: Optional[str] = "low"  # if set, pivot to "Low 9:30" etc.
    open_col: Optional[str] = "open"  # if set, pivot to "Open 9:30" etc. for full OHLC bar data
    max_workers: int = 10  # parallelize by year when > 1; default 10 workers
    within_year_workers: int = 16  # within one year: 16 workers work on 1 chunk at a time (chunk = 16 files); merge in original file order


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


# Session windows for load-and-split: (name, (h, m) start inclusive, (h, m) end inclusive). 16:00 in RTH only.
_SPLIT_SESSION_WINDOWS: list[tuple[str, tuple[int, int], tuple[int, int]]] = [
    ("pm", (4, 0), (9, 29)),
    ("rth", (9, 30), (16, 0)),
    ("ah", (16, 1), (20, 0)),
]


def _column_minute_int(col: str) -> Optional[int]:
    """Parse minute-of-day from a wide cache column (e.g. '9:30', 'Vol 9:30', 'High 16:00'). Returns None for Ticker/Date."""
    if col in ("Ticker", "Date"):
        return None
    part = col
    for prefix in ("Vol ", "High ", "Low ", "Open ", "Close "):
        if col.startswith(prefix):
            part = col[len(prefix) :].strip()
            break
    if ":" not in part:
        return None
    try:
        h, m = part.split(":", 1)
        return int(h) * 60 + int(m)
    except (ValueError, TypeError):
        return None


def _assign_columns_to_sessions(
    all_cols: list[str],
) -> dict[str, list[str]]:
    """Assign each column to exactly one session (pm/rth/ah). Ticker and Date go into all sessions."""
    base = ["Ticker", "Date"]
    out: dict[str, list[str]] = {name: list(base) for name, _, _ in _SPLIT_SESSION_WINDOWS}
    for col in all_cols:
        if col in base:
            continue
        min_int = _column_minute_int(col)
        if min_int is None:
            continue
        for name, start_hm, end_hm in _SPLIT_SESSION_WINDOWS:
            s = start_hm[0] * 60 + start_hm[1]
            e = end_hm[0] * 60 + end_hm[1]
            if s <= min_int <= e:
                out[name].append(col)
                break
    return out


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

    # Ensure all expected columns exist, even if they are entirely NaN for this chunk.
    # This keeps temp chunk schemas identical so streaming merge can use a single schema.
    for col in keep:
        if col not in wide.columns:
            wide[col] = pd.Series(pd.NA, index=wide.index)

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


def _load_one_year_worker(
    year: int, paths: list[str], config: "ParquetLoaderConfig"
) -> tuple[int, Optional[str]]:
    """Load one year in a worker process. Returns (year, cache_path_str). Used when max_workers > 1."""
    loader = ParquetDataLoader(config)
    result = loader._process_one_year(year, paths)
    if result is None:
        return (year, None)
    return (year, str(result))


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

        year_to_paths: dict[int, list[str]] = {}
        for y, p in files_with_year:
            if not os.path.exists(p):
                continue
            if years_filter is not None and y > 0 and y not in years_filter:
                continue
            year_to_paths.setdefault(y, []).append(p)

        years_to_process = sorted(year_to_paths.keys())
        if years_filter is not None:
            years_to_process = [y for y in years_to_process if y in years_filter]
        total_files = sum(len(year_to_paths[y]) for y in years_to_process)
        show_progress = getattr(self.config, "show_progress", True)
        tqdm_fn = _get_tqdm()

        # Single progress bar over all files (0–100% fills as each file/chunk completes), like backtest engine
        pbar = None
        if show_progress and total_files > 0 and tqdm_fn is not None:
            pbar = tqdm_fn(total=total_files, desc="Loading data", unit="file")
        files_done_so_far: list[int] = [0]  # mutable for closure
        last_done_in_year: list[int] = [0]

        def _progress_callback(done_in_year: int, total_in_year: int) -> None:
            if pbar is not None:
                pbar.update(done_in_year - last_done_in_year[0])
                last_done_in_year[0] = done_in_year
            else:
                files_done_so_far[0] += done_in_year - last_done_in_year[0]
                last_done_in_year[0] = done_in_year
                _print_progress_bar(
                    100.0 * files_done_so_far[0] / total_files,
                    files_done_so_far[0],
                    total_files,
                    desc="Loading data",
                )

        out: dict[str, pd.DataFrame] = {}
        for done, year in enumerate(years_to_process):
            if years_filter is not None and year not in years_filter:
                continue
            paths = year_to_paths[year]
            last_done_in_year[0] = 0
            result = self._process_one_year(year, paths, progress_callback=_progress_callback)
            if result is not None:
                out[str(year)] = result
            gc.collect()
            if pbar is None and show_progress and total_files > 0:
                files_done_so_far[0] = sum(len(year_to_paths[y]) for y in years_to_process[: done + 1])
                _print_progress_bar(100.0 * files_done_so_far[0] / total_files, files_done_so_far[0], total_files, desc="Loading data")

        if pbar is not None:
            pbar.close()
        elif show_progress and total_files > 0:
            _print_progress_bar(100.0, total_files, total_files, desc="Loading data")
            import sys
            sys.stdout.write("\n")
            sys.stdout.flush()

        stream_only = getattr(self.config, "stream_from_cache", True)
        if not stream_only:
            for k in list(out.keys()):
                if isinstance(out[k], Path):
                    out[k] = pd.read_parquet(out[k])

        # Only save cache when all requested years succeeded (avoid partial cache)
        requested = {str(y) for y in year_to_paths.keys()}
        if self.config.cache_dir and out and set(out.keys()) == requested:
            self._save_cache_meta(signatures, list(out.keys()), years_filter)
        return out


    def resolve_split_session_cache_paths(
        self,
        cache_dir: str,
        years: list[int],
        *,
        cache_prefix: str = "vwap",
    ) -> dict[str, dict[str, Path]]:
        """Resolve split-session cache paths without loading data.

        Supports both layouts:
        - Subdir: {cache_dir}/pm/{year}.parquet (and rth/ah)
        - Flat legacy: {cache_dir}/{cache_prefix}_pm_{year}.parquet (and rth/ah)

        Returns: {"pm": { "2026": Path(...) }, "rth": {...}, "ah": {...}}
        """
        cache_dir_p = Path(cache_dir)
        by_session: dict[str, dict[str, Path]] = {"pm": {}, "rth": {}, "ah": {}}
        prefix = (cache_prefix or "vwap").strip() or "vwap"

        for year in years:
            ys = str(year)
            for sess in ("pm", "rth", "ah"):
                p_sub = cache_dir_p / sess / f"{year}.parquet"
                # Some users keep prefixed files inside session subfolders:
                # e.g. {cache_dir}/pm/normal_pm_2026.parquet
                p_sub_prefixed = cache_dir_p / sess / f"{prefix}_{sess}_{year}.parquet"
                p_flat = cache_dir_p / f"{prefix}_{sess}_{year}.parquet"
                if p_sub.exists():
                    by_session[sess][ys] = p_sub.resolve()
                elif p_sub_prefixed.exists():
                    by_session[sess][ys] = p_sub_prefixed.resolve()
                elif p_flat.exists():
                    by_session[sess][ys] = p_flat.resolve()

        return by_session

    def _process_path_chunk(
        self, year: int, paths_chunk: list[str]
    ) -> tuple[list[str], Optional[pd.DataFrame], Optional[str]]:
        """Process a subset of paths for one year. Returns (temp_files, accum_df, temp_dir) in same order as paths_chunk."""
        import tempfile

        if not paths_chunk:
            return ([], None, None)

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
        start = getattr(self.config, "session_start", (9, 30))
        end = getattr(self.config, "session_end", (16, 0))
        standard_cols = _standard_minute_columns(start, end)
        chunk_size = max(1, self.config.chunk_size)
        stream_to_disk = getattr(self.config, "stream_to_disk", True)

        wide_chunks = []
        temp_files: list[str] = []
        temp_dir: Optional[str] = None
        accum_df: Optional[pd.DataFrame] = None

        for path in paths_chunk:
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
            wide_one = wide_one.loc[:, ~wide_one.columns.duplicated()]
            wide_chunks.append(wide_one)

            if len(wide_chunks) >= chunk_size:
                batch = pd.concat(wide_chunks, ignore_index=True, copy=False)
                wide_chunks = []
                if stream_to_disk:
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp(prefix="parquet_loader_chunk_")
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
                    temp_dir = tempfile.mkdtemp(prefix="parquet_loader_chunk_")
                tf = os.path.join(temp_dir, f"batch_{len(temp_files)}.parquet")
                batch.to_parquet(tf, index=False)
                temp_files.append(tf)
            else:
                accum_df = batch if accum_df is None else pd.concat([accum_df, batch], ignore_index=True, copy=False)
            del batch
            gc.collect()

        return (temp_files, accum_df, temp_dir)

    def _process_one_year(
        self,
        year: int,
        paths: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[Union[Path, pd.DataFrame]]:
        """Load and process one year; return cache Path (when cache_dir set) or DataFrame, or None on failure.
        progress_callback(done, total) is called as files complete within this year (so bar can fill 0-100%)."""
        import tempfile

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
        start = getattr(self.config, "session_start", (9, 30))
        end = getattr(self.config, "session_end", (16, 0))
        standard_cols = _standard_minute_columns(start, end)
        chunk_size = max(1, self.config.chunk_size)
        stream_to_disk = getattr(self.config, "stream_to_disk", True)

        within_year_workers = min(
            max(1, getattr(self.config, "within_year_workers", 1)),
            len(paths),
        )
        chunk_results: Optional[list[tuple[list[str], Optional[pd.DataFrame], Optional[str]]]] = None
        if within_year_workers <= 1:
            # Sequential: existing loop
            wide_chunks: list[pd.DataFrame] = []
            temp_files: list[str] = []
            temp_dir: str | None = None
            accum_df: Optional[pd.DataFrame] = None

            for file_idx, path in enumerate(paths):
                try:
                    df = pd.read_parquet(path, columns=read_cols)
                except Exception:
                    try:
                        df = pd.read_parquet(path)
                    except Exception:
                        if progress_callback is not None:
                            progress_callback(file_idx + 1, len(paths))
                        continue
                if not required.issubset(df.columns):
                    if progress_callback is not None:
                        progress_callback(file_idx + 1, len(paths))
                    continue
                if vc not in df.columns:
                    df[vc] = 1.0
                ts_et = _parse_polygon_timestamp(df[ts])
                df["_date"] = ts_et.dt.normalize()
                df["_time_str"] = _market_time_str(ts_et)
                df = df.dropna(subset=["_date"])
                if df.empty:
                    if progress_callback is not None:
                        progress_callback(file_idx + 1, len(paths))
                    continue
                df["_year"] = df["_date"].dt.year.astype(int)
                df = df[df["_year"] == year]
                if df.empty:
                    if progress_callback is not None:
                        progress_callback(file_idx + 1, len(paths))
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
                    if progress_callback is not None:
                        progress_callback(file_idx + 1, len(paths))
                    continue
                wide_one["Date"] = pd.to_datetime(wide_one["Date"], errors="coerce").dt.normalize()
                wide_one = wide_one.dropna(subset=["Ticker", "Date"])
                keep_open = bool(getattr(self.config, "open_col", None))
                wide_one = _filter_wide_to_standard_columns(wide_one, standard_cols, keep_open=keep_open)
                wide_one = wide_one.loc[:, ~wide_one.columns.duplicated()]
                wide_chunks.append(wide_one)
                if progress_callback is not None:
                    progress_callback(file_idx + 1, len(paths))

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

        else:
            # 16 workers work on 1 chunk at a time together: chunks of 16 files, each chunk processed by 16 workers in parallel; merge in original file order
            n = within_year_workers
            # Chunks of n files each (one "round" at a time)
            path_chunks = [paths[i : i + n] for i in range(0, len(paths), n)]
            chunk_results = []
            files_done = 0
            for path_chunk in path_chunks:
                # Split this chunk among n workers (order preserved: worker 0 = first files, worker 1 = next, ... so merged result matches source file order)
                n_workers = min(n, len(path_chunk))
                per_worker = (len(path_chunk) + n_workers - 1) // n_workers
                sub_chunks = [
                    path_chunk[i * per_worker : (i + 1) * per_worker]
                    for i in range(n_workers)
                ]
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(self._process_path_chunk, year, sc) for sc in sub_chunks]
                    round_results = [fut.result() for fut in futures]
                chunk_results.extend(round_results)
                files_done += len(path_chunk)
                if progress_callback is not None:
                    progress_callback(files_done, len(paths))
            # Merge in exact same order as source path list (chunk 0 files, then chunk 1 files, ...; within chunk, worker 0 then 1 then ...)
            temp_files = [f for r in chunk_results for f in r[0]]
            accs = [r[1] for r in chunk_results if r[1] is not None]
            accum_df = pd.concat(accs, ignore_index=True, copy=False) if accs else None
            temp_dir = None

        if stream_to_disk and not temp_files:
            return None
        if not stream_to_disk and accum_df is None:
            return None

        cache_dir = self._cache_dir() if self.config.cache_dir else None
        split_sessions = getattr(self.config, "split_sessions", False)
        session_output = getattr(self.config, "session_output", "rth") or "rth"
        if split_sessions and cache_dir:
            merge_path = None  # we write to session subdirs
            merge_paths = {name: cache_dir / name / f"{year}.parquet" for name, _, _ in _SPLIT_SESSION_WINDOWS}
        else:
            merge_paths = {}
            merge_path = (cache_dir / f"{year}.parquet") if cache_dir else (Path(temp_dir) / "merged.parquet" if temp_dir else None)

        if stream_to_disk and temp_files and (merge_path is not None or (split_sessions and merge_paths)):
            if cache_dir and not split_sessions:
                cache_dir.mkdir(parents=True, exist_ok=True)
            show_progress = getattr(self.config, "show_progress", True)
            tqdm_fn = _get_tqdm()
            n_temp = len(temp_files)

            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as e:
                raise RuntimeError(
                    "PyArrow is required to merge cache chunks without loading the full year into memory."
                ) from e

            # First pass: build the union of all column names across chunks.
            all_cols: list[str] = []
            all_cols_set: set[str] = set()
            inspect_iter = (
                tqdm_fn(temp_files, desc="Inspecting chunk schemas", total=n_temp, unit="chunk")
                if (show_progress and n_temp and tqdm_fn)
                else temp_files
            )
            for f in inspect_iter:
                tbl = pq.read_table(f)
                # Deduplicate within this table to avoid silly duplicates.
                col_names = list(dict.fromkeys(tbl.column_names))
                for name in col_names:
                    if name not in all_cols_set:
                        all_cols_set.add(name)
                        all_cols.append(name)
                del tbl
                gc.collect()

            # Ensure a stable, sensible column order: keep the discovery order (Ticker/Date first, then minutes etc.).
            if "Ticker" in all_cols:
                all_cols.insert(0, all_cols.pop(all_cols.index("Ticker")))
            if "Date" in all_cols:
                # Put Date right after Ticker
                all_cols.insert(1, all_cols.pop(all_cols.index("Date")))

            session_columns: dict[str, list[str]] = {}
            if split_sessions and merge_paths:
                session_columns = _assign_columns_to_sessions(all_cols)
                for name in merge_paths:
                    (merge_paths[name].parent).mkdir(parents=True, exist_ok=True)

            # Second pass: stream-write all chunks, padding missing columns with nulls to match all_cols.
            merge_iter = (
                tqdm_fn(temp_files, desc="Merging chunks", total=n_temp, unit="chunk")
                if (show_progress and n_temp and tqdm_fn)
                else temp_files
            )

            # We'll infer field types from the first table and default to float for any extras.
            writer = None
            writers: dict[str, pq.ParquetWriter] = {}
            field_types: dict[str, pa.DataType] = {}
            for f in merge_iter:
                tbl = pq.read_table(f)
                # Align this table to the full column list, adding null columns where missing.
                cols = {}
                for name in all_cols:
                    if name in tbl.column_names:
                        col = tbl[name]
                        cols[name] = col
                        if name not in field_types:
                            field_types[name] = col.type
                    else:
                        # Column missing in this chunk: create a null column.
                        if name not in field_types:
                            field_types[name] = pa.float64()
                        cols[name] = pa.nulls(len(tbl), type=field_types[name])
                aligned = pa.table(cols)

                if split_sessions and merge_paths:
                    for sess_name, sess_cols in session_columns.items():
                        if not sess_cols:
                            continue
                        # Build table with only this session's columns (that exist in aligned).
                        sc = [c for c in sess_cols if c in aligned.column_names]
                        if not sc:
                            continue
                        sess_col = {c: aligned[c] for c in sc}
                        stbl = pa.table(sess_col)
                        if sess_name not in writers:
                            writers[sess_name] = pq.ParquetWriter(str(merge_paths[sess_name]), stbl.schema)
                        writers[sess_name].write_table(stbl)
                else:
                    if writer is None:
                        writer = pq.ParquetWriter(str(merge_path), aligned.schema)
                    writer.write_table(aligned)
                del tbl, aligned, cols
                gc.collect()

            if writer is not None:
                writer.close()
            for w in writers.values():
                w.close()
            if writer is not None or writers:
                if show_progress and tqdm_fn:
                    write_pbar = tqdm_fn(total=1, desc="Writing cache", unit="file")
                    write_pbar.update(1)
                    write_pbar.close()
                elif show_progress and n_temp:
                    _print_progress_bar(100.0, 1, 1, desc="Writing cache", unit="file")
                    import sys
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            if chunk_results is not None:
                for tfs, _tdf, tdir in chunk_results:
                    for f in tfs:
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                    if tdir:
                        try:
                            os.rmdir(tdir)
                        except OSError:
                            pass
            else:
                for f in temp_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                if temp_dir:
                    try:
                        os.rmdir(temp_dir)
                    except OSError:
                        pass
            if cache_dir:
                if split_sessions and merge_paths and session_output in merge_paths and merge_paths[session_output].exists():
                    return merge_paths[session_output]
                if not split_sessions and merge_path is not None and merge_path.exists():
                    return merge_path
            if not split_sessions and merge_path is not None and merge_path.exists():
                return merge_path
            return None
        else:
            result = accum_df
            if cache_dir and result is not None and not result.empty:
                if split_sessions and merge_paths:
                    session_columns = _assign_columns_to_sessions(list(result.columns))
                    session_output_val = getattr(self.config, "session_output", "rth") or "rth"
                    for name, path in merge_paths.items():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        sc = [c for c in session_columns.get(name, []) if c in result.columns]
                        if sc:
                            result[sc].to_parquet(path, index=False)
                    return merge_paths.get(session_output_val)
                cache_dir.mkdir(parents=True, exist_ok=True)
                result.to_parquet(cache_dir / f"{year}.parquet", index=False)
                return cache_dir / f"{year}.parquet"
            if result is not None and not result.empty:
                result = result.drop_duplicates(subset=["Ticker", "Date"], keep="last")
                return result
        return None

    def _cache_dir(self) -> Path:
        """Parquet cache directory (one .parquet per year, _meta.json)."""
        return Path(self.config.cache_dir)

    def _try_load_cache(
        self, signatures: dict[str, tuple[int, float]], years_filter: Optional[set[int]]
    ) -> Optional[dict[str, pd.DataFrame]]:
        cache_dir = self._cache_dir()
        meta_path = cache_dir / "_meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                return None
        else:
            # No meta: try legacy split layout (e.g. vwap_pm_YYYY.parquet, nasdaq_rth_YYYY.parquet in cache_dir)
            import re as _re
            year_re = _re.compile(r"^(?P<prefix>[A-Za-z0-9]+)_(?:pm|rth|ah)_(?P<year>\d{4})\.parquet$")
            legacy_years: set[str] = set()
            legacy_prefixes: set[str] = set()
            if cache_dir.is_dir():
                for p in cache_dir.iterdir():
                    if not (p.is_file() and p.name.lower().endswith(".parquet")):
                        continue
                    m = year_re.match(p.name)
                    if m:
                        legacy_years.add(m.group("year"))
                        legacy_prefixes.add(m.group("prefix"))
            if not legacy_years:
                return None
            # Prefer configured prefix if present in folder, else pick the only one, else default to "normal".
            cfg_prefix = getattr(self.config, "cache_prefix", "normal") or "normal"
            if cfg_prefix in legacy_prefixes:
                chosen_prefix = cfg_prefix
            elif len(legacy_prefixes) == 1:
                chosen_prefix = next(iter(legacy_prefixes))
            else:
                chosen_prefix = "normal"
            meta = {"years": sorted(legacy_years), "split_sessions": True, "cache_prefix": chosen_prefix}
        try:
            # When we have source file list: require cached source files exist and sizes match.
            if signatures and meta:
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
            split_sessions = meta.get("split_sessions", False)
            session_output = getattr(self.config, "session_output", "rth") or "rth"
            cache_prefix = meta.get("cache_prefix") or getattr(self.config, "cache_prefix", "normal") or "normal"
            for year in want_years:
                if year not in expected_years:
                    return None
                if split_sessions:
                    fp = cache_dir / session_output / f"{year}.parquet"
                    if not fp.exists():
                        # Flat legacy split cache: try configured prefix first, then older "vwap_*" for backward compat.
                        fp = cache_dir / f"{cache_prefix}_{session_output}_{year}.parquet"
                        if not fp.exists() and cache_prefix != "vwap":
                            fp2 = cache_dir / f"vwap_{session_output}_{year}.parquet"
                            if fp2.exists():
                                fp = fp2
                else:
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
        split_sessions = getattr(self.config, "split_sessions", False)
        cache_prefix = getattr(self.config, "cache_prefix", "normal") or "normal"
        meta = {
            "signatures": {k: list(v) for k, v in signatures.items()},
            "years": years,
            "years_filter": yf,
            "split_sessions": split_sessions,
            "cache_prefix": cache_prefix,
        }
        with open(cache_dir / "_meta.json", "w") as f:
            json.dump(meta, f)


def resolve_split_session_cache_paths(
    cache_dir: str,
    years: list[int],
    *,
    cache_prefix: str = "vwap",
) -> dict[str, dict[str, Path]]:
    """Resolve split-session cache paths without loading data.

    Supports both layouts:
    - Subdir: {cache_dir}/pm/{year}.parquet (and rth/ah)
    - Flat legacy: {cache_dir}/{cache_prefix}_pm_{year}.parquet (and rth/ah)

    Returns: {"pm": { "2026": Path(...) }, "rth": {...}, "ah": {...}}
    """
    cache_dir_p = Path(cache_dir)
    by_session: dict[str, dict[str, Path]] = {"pm": {}, "rth": {}, "ah": {}}
    prefix = (cache_prefix or "vwap").strip() or "vwap"

    for year in years:
        ys = str(year)
        for sess in ("pm", "rth", "ah"):
            p_sub = cache_dir_p / sess / f"{year}.parquet"
            p_sub_prefixed = cache_dir_p / sess / f"{prefix}_{sess}_{year}.parquet"
            p_flat = cache_dir_p / f"{prefix}_{sess}_{year}.parquet"
            if p_sub.exists():
                by_session[sess][ys] = p_sub.resolve()
            elif p_sub_prefixed.exists():
                by_session[sess][ys] = p_sub_prefixed.resolve()
            elif p_flat.exists():
                by_session[sess][ys] = p_flat.resolve()

    return by_session

