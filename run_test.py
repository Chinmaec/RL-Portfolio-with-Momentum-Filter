try:  
    from .data_loader import load_data_returns   
    from .environment import Portfolio_Env   
    from .agent import PPOAgent, train   
    from .PCA_factors import PCA   
    from .backtest import backtest, print_results, plot_results, sharpe, max_drawdown, annual_return  
except ImportError:   
    from data_loader import load_data_returns  
    from environment import Portfolio_Env   
    from agent import PPOAgent, train   
    from PCA_factors import PCA   
    from backtest import backtest, print_results, plot_results, sharpe, max_drawdown, annual_return   

import gc   
import os  
import time  
from pathlib import Path  
import numpy as np    
import pandas as pd  
import torch   

SEED = 42   
LOOKBACK = 21  
EPISODES = 160 
BATCH_SIZE = 128   
PCA_VAR = 0.85   
NUM_PERIODS = 10   
NUM_FOLDS = 5   
INITIAL_TRAIN_PERIODS = 5   
WARM_START = True  
SHOW_PLOTS = True  

REBALANCING_PERIOD = 1  
TRANSACTION_COST = 0.001   
MIN_HOLDING_PERIOD = 7  
MIN_WEIGHT_CHANGE = 0.02  

CACHE_WORKERS = min(8, max(1, (os.cpu_count() or 12) - 2))  
CACHE_CHUNK_SIZE = 256  

BASE_DIR = Path(__file__).resolve().parent  
CSV_PATH = BASE_DIR / "sample_data.csv"  
OUTPUT_DIR = BASE_DIR / "outputs" / "walk_forward" 

def split_into_10_periods(data: pd.DataFrame) -> list[pd.DataFrame]:   
    sorted_data = data.sort_index()  
    date_index = sorted_data.index   
    date_chunks = np.array_split(date_index, NUM_PERIODS)  
    periods = []  

    for i, chunk in enumerate(date_chunks, start=1):   
        period_df = sorted_data.loc[chunk].copy()   
        periods.append(period_df)   
        start_date = period_df.index.min().date() if len(period_df) > 0 else "NA"  
        end_date = period_df.index.max().date() if len(period_df) > 0 else "NA"   
        print(f"Period {i:>2}/{NUM_PERIODS} | Start: {start_date} | End: {end_date} | Rows: {len(period_df)}")   

    return periods  

def transform_test_with_train_pca(train_r: pd.DataFrame, test_r: pd.DataFrame, eigenvectors: np.ndarray, factor_cols: list[str]) -> pd.DataFrame:  
    train_arr = train_r.to_numpy()  
    test_arr = test_r.to_numpy()   
    mu = train_arr.mean(axis=0)  
    sigma = train_arr.std(axis=0)   
    sigma[sigma == 0.0] = 1.0   
    test_std = (test_arr - mu) / sigma   
    k = len(factor_cols)  
    test_f = pd.DataFrame(test_std @ eigenvectors[:, :k], index=test_r.index, columns=factor_cols)   
    return test_f  

def compute_series_metrics(r: pd.Series) -> dict:   
    r = r.dropna()   
    if r.empty:  
        return {"cumulative_return": np.nan, "annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "calmar": np.nan}   
    cum = (1.0 + r).cumprod()  
    cumulative_return = float(cum.iloc[-1] - 1.0)   
    ann_ret = float(annual_return(r))  
    sr = float(sharpe(r))  
    mdd = float(max_drawdown(cum)) 
    calmar = (ann_ret / abs(mdd)) if mdd < 0 else np.nan   
    return {"cumulative_return": cumulative_return, "annual_return": ann_ret, "sharpe": sr, "max_drawdown": mdd, "calmar": calmar}   

def main() -> None:  
    total_start = time.perf_counter()   
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   

    np.random.seed(SEED)   
    torch.manual_seed(SEED)   
    torch.set_num_threads(8)   

    returns = load_data_returns(csv_path=CSV_PATH, parse_date=True, na_method="fill", fill_value=0.0)  
    returns = returns.iloc[:].copy()  # Sample 1000 day test 

    periods = split_into_10_periods(returns)  
    n_assets = returns.shape[1]  
    ticker_names = list(returns.columns)   

    all_fold_metrics = []  
    all_agent_series = []  
    all_equal_series = []   
    all_ew_bh_series = []   
    all_weights = []   
    prev_checkpoint_path = None   
    prev_state_dim = None   
    prev_action_dim = None   

    for fold in range(1, NUM_FOLDS + 1):   
        train_end_period = INITIAL_TRAIN_PERIODS + (fold - 1)  
        test_period_idx = train_end_period   
        train_periods = periods[:train_end_period]   
        test_period = periods[test_period_idx]   
        train_r = pd.concat(train_periods, axis=0).sort_index()   
        test_r = test_period.sort_index()   

        print(f"\nFold {fold}/{NUM_FOLDS} | Training on periods 1-{train_end_period} | Testing on period {test_period_idx + 1}")  
        print(f"Train window: {train_r.index.min().date()} -> {train_r.index.max().date()} | Rows: {len(train_r)}") 
        print(f"Test  window: {test_r.index.min().date()} -> {test_r.index.max().date()} | Rows: {len(test_r)}") 

        fold_dir = OUTPUT_DIR / f"fold_{fold:02d}"   
        fold_dir.mkdir(parents=True, exist_ok=True)  

        eigenvalues, eigenvectors, train_f = PCA(train_r, variance=PCA_VAR, plot=False, verbose=True)  
        test_f = transform_test_with_train_pca(train_r=train_r, test_r=test_r, eigenvectors=eigenvectors, factor_cols=list(train_f.columns))  
        k = train_f.shape[1]  
        state_dim = LOOKBACK * k   
        action_dim = n_assets   

        use_warm_start = WARM_START and prev_checkpoint_path is not None and prev_state_dim == state_dim and prev_action_dim == action_dim   
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)  

        if use_warm_start:  
            checkpoint = torch.load(prev_checkpoint_path, map_location="cpu")   
            agent.net.load_state_dict(checkpoint["model_state_dict"])   
            agent.opt.load_state_dict(checkpoint["optimizer_state_dict"])  
            print(f"Warm-start enabled from: {prev_checkpoint_path.name}")   
        else:  
            print("Cold-start training for this fold.")  

        train_env = Portfolio_Env(returns=train_r, pca_factors=train_f, lookback=LOOKBACK, rebalance_every=REBALANCING_PERIOD, transaction_cost=TRANSACTION_COST, min_holding_days=MIN_HOLDING_PERIOD, min_weight_change=MIN_WEIGHT_CHANGE, store_history=False, cache_workers=CACHE_WORKERS, cache_chunk_size=CACHE_CHUNK_SIZE)  # Build training env from train-only data and factors.
        train_start = time.perf_counter()  # Start fold training timer.
        train(train_env, agent, n_episodes=EPISODES, batch_size=BATCH_SIZE)   
        train_time = time.perf_counter() - train_start   
        print(f"Fold {fold} training time: {train_time:.2f} sec") 

        checkpoint_path = fold_dir / f"ppo_fold_{fold:02d}.pt"  # Define per-fold model checkpoint path.
        torch.save({"fold": fold, "state_dim": state_dim, "action_dim": action_dim, "model_state_dict": agent.net.state_dict(), "optimizer_state_dict": agent.opt.state_dict()}, checkpoint_path)   
        np.save(fold_dir / f"pca_eigenvalues_fold_{fold:02d}.npy", eigenvalues)  
        np.save(fold_dir / f"pca_eigenvectors_fold_{fold:02d}.npy", eigenvectors)   

        test_env = Portfolio_Env(returns=test_r, pca_factors=test_f, lookback=LOOKBACK, rebalance_every=REBALANCING_PERIOD, transaction_cost=TRANSACTION_COST, min_holding_days=MIN_HOLDING_PERIOD, min_weight_change=MIN_WEIGHT_CHANGE, store_history=True, cache_workers=CACHE_WORKERS, cache_chunk_size=CACHE_CHUNK_SIZE)  # Build test env from unseen test period only.
        test_start = time.perf_counter()  # Start fold backtest timer.
        agent_r, equal_r, ew_bh_r, weights_df = backtest(test_env, agent, test_r)  # Run fold out-of-sample backtest.
        test_time = time.perf_counter() - test_start  # Measure fold backtest runtime.
        print(f"Fold {fold} testing time: {test_time:.2f} sec")  # Print fold testing time.

        fold_returns = pd.concat([agent_r.rename("agent_return"), equal_r.rename("equal_return"), ew_bh_r.rename("ew_bh_return")], axis=1)   
        fold_returns.to_csv(fold_dir / f"fold_{fold:02d}_returns.csv", index=True)  
        weights_df.to_csv(fold_dir / f"fold_{fold:02d}_weights.csv", index=True)   

        agent_m = compute_series_metrics(agent_r)  # Compute fold metrics for agent.
        equal_m = compute_series_metrics(equal_r)  # Compute fold metrics for equal-weight baseline.
        ewbh_m = compute_series_metrics(ew_bh_r)  # Compute fold metrics for EW buy-and-hold baseline.

        fold_metrics = {"fold": fold, "train_period_start": 1, "train_period_end": train_end_period, "test_period": test_period_idx + 1, "train_start_date": str(train_r.index.min().date()), "train_end_date": str(train_r.index.max().date()), "test_start_date": str(test_r.index.min().date()), "test_end_date": str(test_r.index.max().date()), "train_rows": len(train_r), "test_rows": len(test_r), "n_factors": k, "warm_start_used": bool(use_warm_start), "agent_cumulative_return": agent_m["cumulative_return"], "agent_annual_return": agent_m["annual_return"], "agent_sharpe": agent_m["sharpe"], "agent_max_drawdown": agent_m["max_drawdown"], "agent_calmar": agent_m["calmar"], "equal_cumulative_return": equal_m["cumulative_return"], "equal_annual_return": equal_m["annual_return"], "equal_sharpe": equal_m["sharpe"], "equal_max_drawdown": equal_m["max_drawdown"], "equal_calmar": equal_m["calmar"], "ewbh_cumulative_return": ewbh_m["cumulative_return"], "ewbh_annual_return": ewbh_m["annual_return"], "ewbh_sharpe": ewbh_m["sharpe"], "ewbh_max_drawdown": ewbh_m["max_drawdown"], "ewbh_calmar": ewbh_m["calmar"], "train_seconds": train_time, "test_seconds": test_time}  # Build one complete fold metrics record.

        pd.DataFrame([fold_metrics]).to_csv(fold_dir / f"fold_{fold:02d}_metrics.csv", index=False)  
        all_fold_metrics.append(fold_metrics)  
        all_agent_series.append(agent_r)   
        all_equal_series.append(equal_r)   
        all_ew_bh_series.append(ew_bh_r)   
        all_weights.append(weights_df)  

        prev_checkpoint_path = checkpoint_path  # Update warm-start checkpoint pointer for next fold.
        prev_state_dim = state_dim  
        prev_action_dim = action_dim  

        del train_env   
        del test_env   
        gc.collect()   

    all_fold_metrics_df = pd.DataFrame(all_fold_metrics) 
    all_fold_metrics_df.to_csv(OUTPUT_DIR / "all_fold_metrics.csv", index=False)   

    combined_agent = pd.concat(all_agent_series).sort_index()  
    combined_equal = pd.concat(all_equal_series).sort_index()   
    combined_ew_bh = pd.concat(all_ew_bh_series).sort_index()   
    combined_weights = pd.concat(all_weights).sort_index()   

    combined_returns = pd.concat([combined_agent.rename("agent_return"), combined_equal.rename("equal_return"), combined_ew_bh.rename("ew_bh_return")], axis=1)  # Build combined OOS return table.
    combined_returns.to_csv(OUTPUT_DIR / "combined_oos_returns.csv", index=True)  
    combined_weights.to_csv(OUTPUT_DIR / "combined_oos_weights.csv", index=True)   

    print("\n" + "=" * 72) 
    print("FINAL OUT-OF-SAMPLE (CONCATENATED FOLDS 1-5 TEST PERIODS)")   
    print("=" * 72)  # Print separator after heading.
    print_results(combined_agent, combined_equal, combined_ew_bh)   
    plot_results(combined_agent, combined_equal, combined_ew_bh, combined_weights, ticker_names=ticker_names, plot=SHOW_PLOTS)   

    final_agent_metrics = compute_series_metrics(combined_agent)  
    final_equal_metrics = compute_series_metrics(combined_equal)   
    final_ewbh_metrics = compute_series_metrics(combined_ew_bh)   
    final_metrics_df = pd.DataFrame([{"strategy": "agent", **final_agent_metrics}, {"strategy": "equal_weight", **final_equal_metrics}, {"strategy": "ew_buy_hold", **final_ewbh_metrics}])  # Build final metrics table.
    final_metrics_df.to_csv(OUTPUT_DIR / "final_combined_oos_metrics.csv", index=False)   

    total_time = time.perf_counter() - total_start   
    print(f"\nTotal runtime: {total_time:.2f} sec")   
    print(f"Artifacts saved in: {OUTPUT_DIR}")   

if __name__ == "__main__":  # Guard entrypoint to avoid accidental execution on import.
    main()  # Execute end-to-end walk-forward pipeline.
