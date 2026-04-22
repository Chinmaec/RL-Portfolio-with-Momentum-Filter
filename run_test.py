try:  
    from .data_loader import load_data_returns   
    from .environment import Portfolio_Env   
    from .agent import PPOAgent, train   
    from .PCA_factors import PCA   
    from .backtest import backtest, print_results, plot_results, sharpe, max_drawdown, annual_return  
except ImportError:  # Fallback to script-style imports so direct `python run.py` also works.
    from data_loader import load_data_returns  # Fallback local import for data loader.
    from environment import Portfolio_Env  # Fallback local import for environment.
    from agent import PPOAgent, train  # Fallback local import for PPO agent and train loop.
    from PCA_factors import PCA  # Fallback local import for PCA function.
    from backtest import backtest, print_results, plot_results, sharpe, max_drawdown, annual_return  # Fallback local backtest utilities.

import gc  # Import garbage collector to free memory between folds.
import os  # Import os for CPU count and path-safe operations.
import time  # Import time for per-fold timing and runtime transparency.
from pathlib import Path  # Import Path for robust filesystem paths.
import numpy as np  # Import NumPy for numerical work and period splitting.
import pandas as pd  # Import pandas for DataFrame manipulation and result storage.
import torch  # Import torch for seed control and checkpoint save/load.

SEED = 42  # Keep deterministic seed for reproducibility.
LOOKBACK = 21  # Keep original state lookback window.
EPISODES = 120 # Keep original episode count per fold.
BATCH_SIZE = 128  # Keep original PPO batch size.
PCA_VAR = 0.90  # Keep explained-variance threshold for PCA.
NUM_PERIODS = 10  # Split full dataset into exactly 10 chronological periods.
NUM_FOLDS = 5  # Run exactly 5 walk-forward folds.
INITIAL_TRAIN_PERIODS = 5  # Fold 1 trains on periods 1-5 and tests on period 6.
WARM_START = True  # Prefer warm-start across folds when model dimensions match.
SHOW_PLOTS = True  # Keep plotting behavior configurable.

REBALANCING_PERIOD = 1  # Keep existing rebalance cadence.
TRANSACTION_COST = 0.001  # Keep existing transaction cost.
MIN_HOLDING_PERIOD = 7  # Keep existing minimum holding days.
MIN_WEIGHT_CHANGE = 0.03  # Keep existing minimum trade threshold.

CACHE_WORKERS = min(8, max(1, (os.cpu_count() or 12) - 2))  # Keep cache worker logic from current code.
CACHE_CHUNK_SIZE = 256  # Keep cache chunk size from current code.

BASE_DIR = Path(__file__).resolve().parent  # Resolve project directory from current file location.
CSV_PATH = BASE_DIR / "sample_data.csv"  # Keep default sample CSV path.
OUTPUT_DIR = BASE_DIR / "outputs" / "walk_forward"  # Save fold-wise models and results in dedicated folder.

def split_into_10_periods(data: pd.DataFrame) -> list[pd.DataFrame]:  # Define explicit splitter for auditable 10-period slicing.
    sorted_data = data.sort_index()  # Enforce chronological order to avoid any accidental time shuffle.
    date_index = sorted_data.index  # Extract full datetime index used for chronological splitting.
    date_chunks = np.array_split(date_index, NUM_PERIODS)  # Split dates into 10 near-equal chunks without overlap.
    periods = []  # Prepare container for period DataFrames in order.

    for i, chunk in enumerate(date_chunks, start=1):  # Iterate periods as 1..10 for readable logs.
        period_df = sorted_data.loc[chunk].copy()  # Slice exact date chunk to keep boundaries date-driven and explicit.
        periods.append(period_df)  # Store this period in chronological order.
        start_date = period_df.index.min().date() if len(period_df) > 0 else "NA"  # Capture period start date for audit.
        end_date = period_df.index.max().date() if len(period_df) > 0 else "NA"  # Capture period end date for audit.
        print(f"Period {i:>2}/{NUM_PERIODS} | Start: {start_date} | End: {end_date} | Rows: {len(period_df)}")  # Log period boundaries.

    return periods  # Return exactly 10 non-overlapping chronological DataFrames.

def transform_test_with_train_pca(train_r: pd.DataFrame, test_r: pd.DataFrame, eigenvectors: np.ndarray, factor_cols: list[str]) -> pd.DataFrame:  # Transform test only with train stats.
    train_arr = train_r.to_numpy()  # Convert train returns to NumPy for mean/std calculations.
    test_arr = test_r.to_numpy()  # Convert test returns to NumPy for projection.
    mu = train_arr.mean(axis=0)  # Compute feature means on train only to prevent leakage.
    sigma = train_arr.std(axis=0)  # Compute feature std on train only to prevent leakage.
    sigma[sigma == 0.0] = 1.0  # Guard against zero variance columns to keep transform numerically stable.
    test_std = (test_arr - mu) / sigma  # Standardize test data using train parameters only.
    k = len(factor_cols)  # Use same number of components as training factor matrix.
    test_f = pd.DataFrame(test_std @ eigenvectors[:, :k], index=test_r.index, columns=factor_cols)  # Project standardized test data into train PCA basis.
    return test_f  # Return test factor DataFrame aligned to test dates.

def compute_series_metrics(r: pd.Series) -> dict:  # Build small metric helper for per-fold and combined OOS series.
    r = r.dropna()  # Remove NaNs to keep metric math valid.
    if r.empty:  # Handle empty series defensively.
        return {"cumulative_return": np.nan, "annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "calmar": np.nan}  # Return NaNs for empty input.
    cum = (1.0 + r).cumprod()  # Compute cumulative wealth path from daily returns.
    cumulative_return = float(cum.iloc[-1] - 1.0)  # Compute total return over the full evaluation window.
    ann_ret = float(annual_return(r))  # Reuse existing annual return helper for consistency.
    sr = float(sharpe(r))  # Reuse existing Sharpe helper for consistency.
    mdd = float(max_drawdown(cum))  # Compute max drawdown from cumulative curve.
    calmar = (ann_ret / abs(mdd)) if mdd < 0 else np.nan  # Compute Calmar safely when drawdown exists.
    return {"cumulative_return": cumulative_return, "annual_return": ann_ret, "sharpe": sr, "max_drawdown": mdd, "calmar": calmar}  # Return compact metric dict.

def main() -> None:  # Define main executable flow.
    total_start = time.perf_counter()  # Start global timer for total runtime visibility.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists before folds start.

    np.random.seed(SEED)  # Seed NumPy for reproducible randomness.
    torch.manual_seed(SEED)  # Seed torch for reproducible model initialization/training.
    torch.set_num_threads(8)  # Keep explicit CPU thread setting from current script.

    returns = load_data_returns(csv_path=CSV_PATH, parse_date=True, na_method="fill", fill_value=0.0)  # Load returns with your current preprocessing behavior.
    returns = returns.iloc[:].copy()  # Keep existing last-1000-row sample restriction.

    periods = split_into_10_periods(returns)  # Split once up front as required.
    n_assets = returns.shape[1]  # Store action dimension from full dataset columns.
    ticker_names = list(returns.columns)  # Keep ticker labels for plotting later.

    all_fold_metrics = []  # Store one metrics record per fold for diagnostics.
    all_agent_series = []  # Store fold-level out-of-sample agent return series for final concatenation.
    all_equal_series = []  # Store fold-level out-of-sample equal-weight return series.
    all_ew_bh_series = []  # Store fold-level out-of-sample EW buy-and-hold return series.
    all_weights = []  # Store fold-level weight history for optional combined plot.
    prev_checkpoint_path = None  # Track previous fold checkpoint for warm-start.
    prev_state_dim = None  # Track previous fold state dimension for compatibility checks.
    prev_action_dim = None  # Track previous fold action dimension for compatibility checks.

    for fold in range(1, NUM_FOLDS + 1):  # Execute exactly 5 expanding-window folds.
        train_end_period = INITIAL_TRAIN_PERIODS + (fold - 1)  # Compute train end as 5,6,7,8,9 across folds.
        test_period_idx = train_end_period  # Compute test period index as 6,7,8,9,10 in 1-based terms.
        train_periods = periods[:train_end_period]  # Build expanding training window from period 1 forward.
        test_period = periods[test_period_idx]  # Select exactly one unseen test period for this fold.
        train_r = pd.concat(train_periods, axis=0).sort_index()  # Concatenate train periods and enforce chronological order.
        test_r = test_period.sort_index()  # Enforce chronological order for test period.

        print(f"\nFold {fold}/{NUM_FOLDS} | Training on periods 1-{train_end_period} | Testing on period {test_period_idx + 1}")  # Print transparent fold progress.
        print(f"Train window: {train_r.index.min().date()} -> {train_r.index.max().date()} | Rows: {len(train_r)}")  # Log train boundaries for auditability.
        print(f"Test  window: {test_r.index.min().date()} -> {test_r.index.max().date()} | Rows: {len(test_r)}")  # Log test boundaries for auditability.

        fold_dir = OUTPUT_DIR / f"fold_{fold:02d}"  # Create fold-specific directory for crash-safe outputs.
        fold_dir.mkdir(parents=True, exist_ok=True)  # Ensure fold directory exists before training begins.

        eigenvalues, eigenvectors, train_f = PCA(train_r, variance=PCA_VAR, plot=False, verbose=True)  # Fit PCA only on training data for this fold.
        test_f = transform_test_with_train_pca(train_r=train_r, test_r=test_r, eigenvectors=eigenvectors, factor_cols=list(train_f.columns))  # Transform test with train-only stats.
        k = train_f.shape[1]  # Read fold-specific number of PCA factors retained.
        state_dim = LOOKBACK * k  # Build state dimension from lookback and retained factors.
        action_dim = n_assets  # Keep action dimension as number of assets.

        use_warm_start = WARM_START and prev_checkpoint_path is not None and prev_state_dim == state_dim and prev_action_dim == action_dim  # Warm-start only when prior checkpoint exists and dimensions match.
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)  # Always create a fresh agent object for clean fold isolation.

        if use_warm_start:  # Load previous fold weights when warm-start is safe.
            checkpoint = torch.load(prev_checkpoint_path, map_location="cpu")  # Load previous fold checkpoint.
            agent.net.load_state_dict(checkpoint["model_state_dict"])  # Restore policy/value network weights.
            agent.opt.load_state_dict(checkpoint["optimizer_state_dict"])  # Restore optimizer state for smoother continuation.
            print(f"Warm-start enabled from: {prev_checkpoint_path.name}")  # Log warm-start source checkpoint.
        else:  # Start from scratch when first fold or incompatible dimensions.
            print("Cold-start training for this fold.")  # Log cold-start reason clearly.

        train_env = Portfolio_Env(returns=train_r, pca_factors=train_f, lookback=LOOKBACK, rebalance_every=REBALANCING_PERIOD, transaction_cost=TRANSACTION_COST, min_holding_days=MIN_HOLDING_PERIOD, min_weight_change=MIN_WEIGHT_CHANGE, store_history=False, cache_workers=CACHE_WORKERS, cache_chunk_size=CACHE_CHUNK_SIZE)  # Build training env from train-only data and factors.
        train_start = time.perf_counter()  # Start fold training timer.
        train(train_env, agent, n_episodes=EPISODES, batch_size=BATCH_SIZE)  # Train PPO on this fold's expanding training window.
        train_time = time.perf_counter() - train_start  # Measure training runtime for fold diagnostics.
        print(f"Fold {fold} training time: {train_time:.2f} sec")  # Print fold training time.

        checkpoint_path = fold_dir / f"ppo_fold_{fold:02d}.pt"  # Define per-fold model checkpoint path.
        torch.save({"fold": fold, "state_dim": state_dim, "action_dim": action_dim, "model_state_dict": agent.net.state_dict(), "optimizer_state_dict": agent.opt.state_dict()}, checkpoint_path)  # Save fold model so prior folds survive crashes.
        np.save(fold_dir / f"pca_eigenvalues_fold_{fold:02d}.npy", eigenvalues)  # Save fold PCA eigenvalues for reproducibility.
        np.save(fold_dir / f"pca_eigenvectors_fold_{fold:02d}.npy", eigenvectors)  # Save fold PCA eigenvectors for reproducibility.

        test_env = Portfolio_Env(returns=test_r, pca_factors=test_f, lookback=LOOKBACK, rebalance_every=REBALANCING_PERIOD, transaction_cost=TRANSACTION_COST, min_holding_days=MIN_HOLDING_PERIOD, min_weight_change=MIN_WEIGHT_CHANGE, store_history=True, cache_workers=CACHE_WORKERS, cache_chunk_size=CACHE_CHUNK_SIZE)  # Build test env from unseen test period only.
        test_start = time.perf_counter()  # Start fold backtest timer.
        agent_r, equal_r, ew_bh_r, weights_df = backtest(test_env, agent, test_r)  # Run fold out-of-sample backtest.
        test_time = time.perf_counter() - test_start  # Measure fold backtest runtime.
        print(f"Fold {fold} testing time: {test_time:.2f} sec")  # Print fold testing time.

        fold_returns = pd.concat([agent_r.rename("agent_return"), equal_r.rename("equal_return"), ew_bh_r.rename("ew_bh_return")], axis=1)  # Combine fold return series in one table.
        fold_returns.to_csv(fold_dir / f"fold_{fold:02d}_returns.csv", index=True)  # Save fold returns for crash-safe recovery and audit.
        weights_df.to_csv(fold_dir / f"fold_{fold:02d}_weights.csv", index=True)  # Save fold weights so allocation path is preserved.

        agent_m = compute_series_metrics(agent_r)  # Compute fold metrics for agent.
        equal_m = compute_series_metrics(equal_r)  # Compute fold metrics for equal-weight baseline.
        ewbh_m = compute_series_metrics(ew_bh_r)  # Compute fold metrics for EW buy-and-hold baseline.

        fold_metrics = {"fold": fold, "train_period_start": 1, "train_period_end": train_end_period, "test_period": test_period_idx + 1, "train_start_date": str(train_r.index.min().date()), "train_end_date": str(train_r.index.max().date()), "test_start_date": str(test_r.index.min().date()), "test_end_date": str(test_r.index.max().date()), "train_rows": len(train_r), "test_rows": len(test_r), "n_factors": k, "warm_start_used": bool(use_warm_start), "agent_cumulative_return": agent_m["cumulative_return"], "agent_annual_return": agent_m["annual_return"], "agent_sharpe": agent_m["sharpe"], "agent_max_drawdown": agent_m["max_drawdown"], "agent_calmar": agent_m["calmar"], "equal_cumulative_return": equal_m["cumulative_return"], "equal_annual_return": equal_m["annual_return"], "equal_sharpe": equal_m["sharpe"], "equal_max_drawdown": equal_m["max_drawdown"], "equal_calmar": equal_m["calmar"], "ewbh_cumulative_return": ewbh_m["cumulative_return"], "ewbh_annual_return": ewbh_m["annual_return"], "ewbh_sharpe": ewbh_m["sharpe"], "ewbh_max_drawdown": ewbh_m["max_drawdown"], "ewbh_calmar": ewbh_m["calmar"], "train_seconds": train_time, "test_seconds": test_time}  # Build one complete fold metrics record.

        pd.DataFrame([fold_metrics]).to_csv(fold_dir / f"fold_{fold:02d}_metrics.csv", index=False)  # Save fold metrics separately so each fold is independently recoverable.
        all_fold_metrics.append(fold_metrics)  # Add fold metrics to in-memory summary list.
        all_agent_series.append(agent_r)  # Add fold agent series for final OOS concatenation.
        all_equal_series.append(equal_r)  # Add fold equal-weight series for final OOS concatenation.
        all_ew_bh_series.append(ew_bh_r)  # Add fold EW buy-and-hold series for final OOS concatenation.
        all_weights.append(weights_df)  # Add fold weight path for optional combined plot.

        prev_checkpoint_path = checkpoint_path  # Update warm-start checkpoint pointer for next fold.
        prev_state_dim = state_dim  # Store current state dimension for next-fold compatibility check.
        prev_action_dim = action_dim  # Store current action dimension for next-fold compatibility check.

        del train_env  # Release training environment memory after fold completion.
        del test_env  # Release test environment memory after fold completion.
        gc.collect()  # Force garbage collection to reduce memory pressure between folds.

    all_fold_metrics_df = pd.DataFrame(all_fold_metrics)  # Convert fold metrics list to DataFrame for reporting.
    all_fold_metrics_df.to_csv(OUTPUT_DIR / "all_fold_metrics.csv", index=False)  # Save all-fold metrics table.

    combined_agent = pd.concat(all_agent_series).sort_index()  # Concatenate agent OOS fold series in chronological order.
    combined_equal = pd.concat(all_equal_series).sort_index()  # Concatenate equal-weight OOS fold series in chronological order.
    combined_ew_bh = pd.concat(all_ew_bh_series).sort_index()  # Concatenate EW buy-and-hold OOS fold series in chronological order.
    combined_weights = pd.concat(all_weights).sort_index()  # Concatenate fold weights in chronological order.

    combined_returns = pd.concat([combined_agent.rename("agent_return"), combined_equal.rename("equal_return"), combined_ew_bh.rename("ew_bh_return")], axis=1)  # Build combined OOS return table.
    combined_returns.to_csv(OUTPUT_DIR / "combined_oos_returns.csv", index=True)  # Save final concatenated OOS returns.
    combined_weights.to_csv(OUTPUT_DIR / "combined_oos_weights.csv", index=True)  # Save final concatenated OOS weights.

    print("\n" + "=" * 72)  # Print separator before final OOS report.
    print("FINAL OUT-OF-SAMPLE (CONCATENATED FOLDS 1-5 TEST PERIODS)")  # Print final OOS heading.
    print("=" * 72)  # Print separator after heading.
    print_results(combined_agent, combined_equal, combined_ew_bh)  # Reuse existing result printer on combined OOS series.
    plot_results(combined_agent, combined_equal, combined_ew_bh, combined_weights, ticker_names=ticker_names, plot=SHOW_PLOTS)  # Reuse existing plotter on combined OOS series.

    final_agent_metrics = compute_series_metrics(combined_agent)  # Compute explicit final agent metrics dictionary.
    final_equal_metrics = compute_series_metrics(combined_equal)  # Compute explicit final equal-weight metrics dictionary.
    final_ewbh_metrics = compute_series_metrics(combined_ew_bh)  # Compute explicit final EW buy-and-hold metrics dictionary.
    final_metrics_df = pd.DataFrame([{"strategy": "agent", **final_agent_metrics}, {"strategy": "equal_weight", **final_equal_metrics}, {"strategy": "ew_buy_hold", **final_ewbh_metrics}])  # Build final metrics table.
    final_metrics_df.to_csv(OUTPUT_DIR / "final_combined_oos_metrics.csv", index=False)  # Save final OOS metrics table.

    total_time = time.perf_counter() - total_start  # Compute full pipeline runtime.
    print(f"\nTotal walk-forward runtime: {total_time:.2f} sec")  # Print total runtime for transparency.
    print(f"Artifacts saved in: {OUTPUT_DIR}")  # Print output path so results are easy to locate.

if __name__ == "__main__":  # Guard entrypoint to avoid accidental execution on import.
    main()  # Execute end-to-end walk-forward pipeline.
