from data_loader import load_data_returns
from environment import Portfolio_Env
from agent import PPOAgent, train
from PCA_factors import PCA
from backtest import plot_results, print_results, backtest


import pandas as pd
import numpy as np
from pathlib import Path
import torch
import time 
import os 

SEED = 42
LOOKBACK = 21
TRAIN_SPLIT = 0.70
EPISODES = 160
BATCH_SIZE = 128
PCA_VAR = 0.90

REBALANCING_PERIOD = 1
TRANSACTION_COST = 0.001
MIN_HOLDING_PERIOD = 7
MIN_WEIGHT_CHANGE = 0.03

CACHE_WORKERS = min(8, max(1, (os.cpu_count() or 12) - 1))
CACHE_CHUNK_SIZE = 256
SHOW_PLOTS = True

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sample_data.csv"      
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    total_start = time.perf_counter()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)

    # Load your data
    returns = load_data_returns(
        csv_path=CSV_PATH,
        parse_date=True,
        na_method="fill",   # "drop" or "fill"
        fill_value=0.0
    )

    # Split train/test data
    T, N = returns.shape
    split = int(TRAIN_SPLIT * T)
    train_r = returns.iloc[:split]
    test_r = returns.iloc[split:]

    # Fit PCA on train data
    eigenvalues, eigenvectors, train_f = PCA(
        train_r, variance=PCA_VAR, plot=False, verbose=True
    )

    train_arr = train_r.to_numpy()
    test_arr = test_r.to_numpy()

    mu = train_arr.mean(axis=0)
    sigma = train_arr.std(axis=0)
    sigma[sigma == 0.0] = 1.0

    test_std = (test_arr - mu) / sigma
    K = train_f.shape[1]

    test_f = pd.DataFrame(
        test_std @ eigenvectors[:, :K],
        index=test_r.index,
        columns=train_f.columns
    )

    ticker_names = list(returns.columns)
    state_dim = LOOKBACK * K 
    action_dim = N

    # Train
    print("Training agent...")
    train_env = Portfolio_Env(
        train_r,
        train_f,
        lookback=LOOKBACK,
        rebalance_every=REBALANCING_PERIOD,
        transaction_cost=TRANSACTION_COST,
        min_holding_days=MIN_HOLDING_PERIOD,
        min_weight_change=MIN_WEIGHT_CHANGE,
        store_history=False,
        cache_workers=CACHE_WORKERS,
        cache_chunk_size=CACHE_CHUNK_SIZE,)   
    
    agent = PPOAgent(state_dim, action_dim)

    # train(train_env, agent, n_episodes=EPISODES, batch_size=BATCH_SIZE)
    train_start = time.perf_counter()
    train(train_env, agent, n_episodes=EPISODES, batch_size=BATCH_SIZE)
    train_time = time.perf_counter() - train_start
    print(f"Training time: {train_time:.2f} sec")


    # Backtest on unseen test data
    print("\nBacktesting on test set...")
    test_env = Portfolio_Env(
    test_r,
    test_f,
    lookback=LOOKBACK,
    rebalance_every=REBALANCING_PERIOD,
    transaction_cost=TRANSACTION_COST,
    min_holding_days=MIN_HOLDING_PERIOD,
    min_weight_change=MIN_WEIGHT_CHANGE,
    store_history=True,
    cache_workers=CACHE_WORKERS,
    cache_chunk_size=CACHE_CHUNK_SIZE,
    )  

    test_start = time.perf_counter()
    agent_r, equal_r, weights_df = backtest(test_env, agent)
    test_time = time.perf_counter() - test_start
    print(f"Testing time: {test_time:.2f} sec")
    
    # Results
    print_results(agent_r, equal_r)
    plot_results(agent_r, equal_r, weights_df, ticker_names, plot = SHOW_PLOTS)
    

if __name__ == "__main__":
    main()
