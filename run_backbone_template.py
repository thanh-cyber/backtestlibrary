from datetime import time

from backtestlibrary import (
    BacktestConfig,
    ChronologicalBacktestEngine,
    GapperDataLoader,
    LoaderConfig,
    build_full_metrics,
    run_monte_carlo,
)
from backtestlibrary.strategy_template import StrategyTemplate


def main() -> None:
    loader = GapperDataLoader(
        LoaderConfig(
            gapper_dir=r"C:\Users\johnn\stock_data_backtest\gapper",
            cache_dir=r"C:\Users\johnn\stock_data_backtest\cache\backbone_cleaned_year_data",
        )
    )
    cleaned_year_data = loader.load_cleaned_year_data()

    engine = ChronologicalBacktestEngine(
        BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
        )
    )

    starting_accounts = [2000, 10000, 50000]
    strategy = StrategyTemplate()  # Replace with your own strategy class
    results = engine.run(cleaned_year_data, strategy, starting_accounts)

    metrics_df, _equity_curves = build_full_metrics(results, starting_accounts)
    mc_df, _mc_cache = run_monte_carlo(results, starting_accounts, mc_runs=5000, seed=42)

    print("Metrics rows:", len(metrics_df))
    print("Monte Carlo rows:", len(mc_df))


if __name__ == "__main__":
    main()
