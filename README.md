# PCA-PPO-Portfolio-Optimization-with-Momentum-Filter-
This project trains a PPO reinforcement learning agent to optimize portfolio weights using PCA-derived market factors as state features.   The agent is trained to outperform an equal-weight benchmark under transaction costs and trading constraints.


## Motivation
Classical portfolio methods are often static. This project explores a dynamic allocation approach where:
- State = rolling PCA factors from asset returns
- Policy = PPO agent outputs portfolio weights
- Overlay = momentum-based filter to suppress weak assets
- Objective = beat equal-weight benchmark net of costs

## Method Overview
1. Load price data and convert to returns.
2. Split train/test data.
3. Fit PCA on train returns and project both train/test into factor space.
4. Train PPO agent in custom environment:
   - Rebalancing frequency
   - Transaction cost penalty
   - Minimum holding period
   - Minimum weight-change threshold
5. Backtest on unseen test data vs equal-weight portfolio.

## Project Structure
- `run.py` - end-to-end training and backtest entrypoint
- `data.py` - loading and return construction
- `factors.py` - PCA feature extraction
- `environment.py` - RL environment and reward logic
- `agent.py` - PPO policy/critic and training loop
- `backtest.py` - performance metrics and plots
- `sample_data.csv` - sample dataset
- `making_sample_data.ipynb` - data preparation notebook

## Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
