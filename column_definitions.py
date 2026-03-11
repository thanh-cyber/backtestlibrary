"""
Canonical entry, exit, and continuous column lists for backtestlibrary.

These are the columns written to the entry CSV (Entry_Col_*), exit CSV (Exit_Col_*),
and continuous CSV (Continuous_Col_*_Entry/Exit/Max/Min/At30min/At60min) when
split_entry_exit=True. Projects can override by defining ENTRY_COLUMNS,
EXIT_SNAPSHOT_COLUMNS, and CONTINUOUS_TRACKING_COLUMNS in their column_library module.
"""

from __future__ import annotations

from typing import List

# ----- Entry columns: snapshot at entry bar (one CSV column per item) -----
ENTRY_COLUMNS: List[str] = [
    # Volatility
    "Col_ATR14",
    "Col_VWAP",  # often injected from wide prefill
    "Col_NormalizedATR_Pct",
    "Col_ATR_vs_20dayAvg_Pct",
    "Col_HistoricalVol_20day",
    "Col_TrueRange_vs_ATR",
    "Col_DailyRange_vs_ATR_Pct",
    # Trend / momentum
    "Col_DistTo50MA_ATR",
    "Col_DistTo200MA_ATR",
    "Col_PriceAbove200MA_ATR",
    "Col_ADX14",
    "Col_DI_Diff",
    "Col_MACD_Hist",
    "Col_MACDV_Normalized",
    "Col_ROC10",
    "Col_ROC20",
    "Col_20dayLinReg_Slope_ATR",
    # Oscillators
    "Col_RSI14",
    "Col_StochK_14_3",
    "Col_StochD",
    "Col_CCI20",
    "Col_BollingerPctB",
    "Col_DistUpperBB_ATR",
    "Col_DistLowerBB_ATR",
    # Volume / VWAP
    "Col_RelativeVolume",
    "Col_TodayVol_vs_YestVol",
    "Col_OBV_Slope5",
    "Col_AccumDist",
    "Col_VWAP_Deviation_ATR",
    "Col_VWAP_Deviation_Pct",
    "Col_VWAP_AbsExtension_ATR",
    "Col_VWAP_Dist_1SD_ATR",
    "Col_VWAP_Dist_m1SD_ATR",
    "Col_VWAP_Dist_2SD_ATR",
    "Col_VWAP_Dist_m2SD_ATR",
    "Col_VWAP_SD_Multiples",
    "Col_VWAP_Slope10_ATR",
    "Col_VWAP_ROC5",
    "Col_VWAP_PrevDayClose_Dist_ATR",
    "Col_VWAP_WeeklyAnchored_Dist_ATR",
    "Col_VWAP_Swing20_Dist_ATR",
    "Col_VWAP_GapOpen_Dist_ATR",
    "Col_BarsSinceVWAP_Cross",
    "Col_VWAP_vs_Open_ATR",
    "Col_VWAP_PosIn2SD_Bands_Pct",
    # Price action
    "Col_PctInYesterdayRange",
    "Col_PctIn5DayRange",
    "Col_DistYesterdayHigh_ATR",
    "Col_DistYesterdayLow_ATR",
    "Col_Dist52wHigh_ATR",
    "Col_Dist52wLow_ATR",
    "Col_OpenToClose_Pct_Sofar",
    "Col_CandleBody_vs_ATR",
    # Gaps
    "Col_Gap_Pct",
    "Col_Gap_ATR",
    "Col_PreMarketGap_ATR",
    "Col_GapFillProxy_ATR",
    # Key levels
    "Col_DistNearestRound_ATR",
    "Col_DistPivotR1S1_ATR",
    "Col_DistSwingHigh5_ATR",
    "Col_DistSwingLow5_ATR",
    # Market context
    "Col_StockVsSPX_TodayPct",
    "Col_RelStrengthVsSector_20d",
    "Col_Beta60d",
    "Col_CorrToSPY_10d",
    # Risk / intra-trade (if present at entry)
    "Col_DistToInitialStop_R",
    "Col_UnrealizedPL_Noon",
    "Col_UnrealizedPL_2pm",
    "Col_UnrealizedPL_30minBeforeClose",
    "Col_MaxFavorableExcursion_R",
    "Col_BarsSinceEntry",
    "Col_PosSize_PctAccount",
    # Time
    "Col_EntryTime_HourNumeric",
    "Col_DayOfWeek",
    "Col_SessionFlag",
]

# ----- Exit columns: snapshot at exit bar (one CSV column per item) -----
EXIT_SNAPSHOT_COLUMNS: List[str] = [
    # Volatility
    "Col_ATR14",
    "Col_VWAP",  # often injected from wide prefill
    "Col_NormalizedATR_Pct",
    "Col_ATR_vs_20dayAvg_Pct",
    "Col_HistoricalVol_20day",
    "Col_TrueRange_vs_ATR",
    "Col_DailyRange_vs_ATR_Pct",
    # Trend / momentum
    "Col_DistTo50MA_ATR",
    "Col_DistTo200MA_ATR",
    "Col_PriceAbove200MA_ATR",
    "Col_ADX14",
    "Col_DI_Diff",
    "Col_MACD_Hist",
    "Col_MACDV_Normalized",
    "Col_ROC10",
    "Col_ROC20",
    "Col_20dayLinReg_Slope_ATR",
    # Oscillators
    "Col_RSI14",
    "Col_StochK_14_3",
    "Col_StochD",
    "Col_CCI20",
    "Col_BollingerPctB",
    "Col_DistUpperBB_ATR",
    "Col_DistLowerBB_ATR",
    # Volume / VWAP
    "Col_RelativeVolume",
    "Col_TodayVol_vs_YestVol",
    "Col_OBV_Slope5",
    "Col_AccumDist",
    "Col_VWAP_Deviation_ATR",
    "Col_VWAP_Deviation_Pct",
    "Col_VWAP_AbsExtension_ATR",
    "Col_VWAP_Dist_1SD_ATR",
    "Col_VWAP_Dist_m1SD_ATR",
    "Col_VWAP_Dist_2SD_ATR",
    "Col_VWAP_Dist_m2SD_ATR",
    "Col_VWAP_SD_Multiples",
    "Col_VWAP_Slope10_ATR",
    "Col_VWAP_ROC5",
    "Col_VWAP_PrevDayClose_Dist_ATR",
    "Col_VWAP_WeeklyAnchored_Dist_ATR",
    "Col_VWAP_Swing20_Dist_ATR",
    "Col_VWAP_GapOpen_Dist_ATR",
    "Col_BarsSinceVWAP_Cross",
    "Col_VWAP_vs_Open_ATR",
    "Col_VWAP_PosIn2SD_Bands_Pct",
    # Price action
    "Col_PctInYesterdayRange",
    "Col_PctIn5DayRange",
    "Col_DistYesterdayHigh_ATR",
    "Col_DistYesterdayLow_ATR",
    "Col_Dist52wHigh_ATR",
    "Col_Dist52wLow_ATR",
    "Col_OpenToClose_Pct_Sofar",
    "Col_CandleBody_vs_ATR",
    # Gaps
    "Col_Gap_Pct",
    "Col_Gap_ATR",
    "Col_PreMarketGap_ATR",
    "Col_GapFillProxy_ATR",
    # Key levels
    "Col_DistNearestRound_ATR",
    "Col_DistPivotR1S1_ATR",
    "Col_DistSwingHigh5_ATR",
    "Col_DistSwingLow5_ATR",
    # Market context
    "Col_StockVsSPX_TodayPct",
    "Col_RelStrengthVsSector_20d",
    "Col_Beta60d",
    "Col_CorrToSPY_10d",
    # Risk / intra-trade at exit
    "Col_DistToInitialStop_R",
    "Col_UnrealizedPL_Noon",
    "Col_UnrealizedPL_2pm",
    "Col_UnrealizedPL_30minBeforeClose",
    "Col_MaxFavorableExcursion_R",
    "Col_BarsSinceEntry",
    "Col_PosSize_PctAccount",
    # Time
    "Col_EntryTime_HourNumeric",
    "Col_DayOfWeek",
    "Col_SessionFlag",
]

# ----- Continuous columns: tracked during trade (6 CSV columns per item: Entry, Exit, Max, Min, At30min, At60min) -----
# Synced with librarycolumn (C:\Users\johnn\librarycolumn) column_library.CONTINUOUS_TRACKING_COLUMNS (20 columns).
CONTINUOUS_TRACKING_COLUMNS: List[str] = [
    "Col_ExtensionFromDaily9EMA_ATR",
    "Col_VWAP_Deviation_ATR",
    "Col_DistToVWAP_Slope10_ATR",
    "Col_DistFromSessionVWAP_ATR",
    "Col_MomentumDivergence_RSI",
    "Col_RSI14",
    "Col_StochK_14_3",
    "Col_MACD_Hist",
    "Col_BollingerPctB",
    "Col_ORB_15min_DistHigh_ATR",
    "Col_StdDev_Last10Bars_ATR",
    "Col_CCI20",
    "Col_VolumeSurge_1min_Ratio",
    "Col_IntradayATR_Ratio",
    "Col_ChoppinessIndex_14",
    "Col_SuperTrend",
    "Col_Keltner_Upper",
    "Col_Keltner_Lower",
    "Col_RelativeVigorIndex",
    "Col_SchaffTrendCycle",
]
