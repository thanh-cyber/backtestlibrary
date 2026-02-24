from datetime import time

from backtestlibrary import (
    BacktestConfig,
    ChronologicalBacktestEngine,
    GapperDataLoader,
    LoaderConfig,
    run_monte_carlo,
)
from backtestlibrary.analyzers import DEFAULT_ANALYZERS
from backtestlibrary.strategy_template import StrategyTemplate


def main() -> None:
    # Option A: backtestdata 5% PM gapper (single price per minute, Vol 4:00, etc.)
    # from backtestdata import load_cleaned_year_data_gapper_5pm
    # cleaned_year_data = load_cleaned_year_data_gapper_5pm(years=[2022], cache_dir="./cache/gapper_5pm")

    # Option B: backtestlibrary GapperDataLoader (gapper_dir can point at backtestdata/gapper)
    loader = GapperDataLoader(
        LoaderConfig(
            gapper_dir=r"C:\Users\johnn\backtestdata\gapper",
            cache_dir=r"C:\Users\johnn\stock_data_backtest\cache\backbone_cleaned_year_data",
        )
    )
    cleaned_year_data = loader.load_cleaned_year_data()

    engine = ChronologicalBacktestEngine(
        BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
        ),
        analyzers=DEFAULT_ANALYZERS,
    )

    starting_accounts = [2000, 10000, 50000]
    strategy = StrategyTemplate()  # Replace with your own strategy class
    results, metrics_df, _equity_curves = engine.run(cleaned_year_data, strategy, starting_accounts)

    mc_df, _mc_cache = run_monte_carlo(results, starting_accounts, mc_runs=5000, seed=42)

    print("Metrics rows:", len(metrics_df))
    print("Monte Carlo rows:", len(mc_df))


if __name__ == "__main__":
    main()
