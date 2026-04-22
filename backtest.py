import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import matplotlib.gridspec as gridspec

# Equally weighted buy and hold benchmark
def ew_buy_and_hold_returns(returns_df, start_idx):
    arr = returns_df.to_numpy()
    n = arr.shape[1]
    w = np.ones(n) / n
    out = []

    for t in range(start_idx, arr.shape[0]):
        r = arr[t]
        day_ret = float(np.dot(w, r))
        out.append(day_ret)

        post = w * (1.0 + r)
        denom = post.sum()
        if denom > 0:
            w = post / denom

    return pd.Series(out, index=returns_df.index[start_idx:])

# Metrics 

def sharpe(returns, periods=252):
    vol = returns.std()
    if vol == 0 or np.isnan(vol):
        return 0.0    
    return (returns.mean() / returns.std()) * np.sqrt(periods)

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    return ((cum_returns - peak) / peak).min()

def annual_return(returns, periods=252):
    return returns.mean() * periods

def cagr(cum_returns, periods=252):
    n_periods = len(cum_returns)
    total_return = cum_returns.iloc[-1]
    years = n_periods / periods
    return total_return ** (1 / years) - 1

def calmar_ratio(cum_returns):
    mdd = max_drawdown(cum_returns)
    return cagr(cum_returns) / abs(mdd)

def annual_vol(returns, periods=252):
    return returns.std() * np.sqrt(periods)

def hit_rate(agent_r,equal_r):
    return (agent_r > equal_r).mean()


# Backtest Functions

def backtest(env, agent, returns_df):
    """
    Run the trained agent on the environment (should be test set).
    Records daily returns and weights for analysis.
    """
    state = env.reset()
    done  = False

    agent_returns  = []
    equal_returns  = []
    weight_history = []

    while not done:
        action = None
        if env.should_rebalance_today():
            action, _, _ = agent.act(state, deterministic=True)

        state, reward, done = env.step(action)

        last = env.last_info
        agent_returns.append(last["port_return"])   # net of transaction cost
        equal_returns.append(last["equal_return"])
        weight_history.append(last["weights"])

    idx = returns_df.index[env.lookback:]
    agent_r = pd.Series(agent_returns, index=idx)
    equal_r = pd.Series(equal_returns, index=idx)
    ew_bh_r = ew_buy_and_hold_returns(returns_df, env.lookback)
    weights_df = pd.DataFrame(weight_history, index=idx)

    # agent_r = pd.Series(agent_returns)
    # equal_r = pd.Series(equal_returns)
    # weights_df = pd.DataFrame(weight_history)

    return agent_r, equal_r, ew_bh_r, weights_df


# Results Table 

def print_results(agent_r, equal_r, ew_bh_r):
    agent_cum = (1 + agent_r).cumprod()
    equal_cum = (1 + equal_r).cumprod()
    ew_bh_cum = (1 + ew_bh_r).cumprod()

    # agent_hit = hit_rate(agent_r, equal_r)
    # equal_hit = (equal_r > agent_r).mean()  
    # ew_bh_hit = " "

    metrics = {
        "Final Value ($1)"  : [agent_cum.iloc[-1],       equal_cum.iloc[-1],       ew_bh_cum.iloc[-1]],
        "Ann. Return"       : [annual_return(agent_r),   annual_return(equal_r),   annual_return(ew_bh_r)],
        "Ann. Volatility"   : [annual_vol(agent_r),      annual_vol(equal_r),      annual_vol(ew_bh_r)],
        "Sharpe Ratio"      : [sharpe(agent_r),          sharpe(equal_r),          sharpe(ew_bh_r)],
        "Max Drawdown"      : [max_drawdown(agent_cum),  max_drawdown(equal_cum),  max_drawdown(ew_bh_cum)],
        "Calmar Ratio"      : [calmar_ratio(agent_cum),  calmar_ratio(equal_cum),  calmar_ratio(ew_bh_cum)]
        # "Hit Rate"          : [agent_hit,                equal_hit,                ew_bh_hit],
    }

    print("\n" + "=" * 66)
    print(f"{'Metric':<22} {'Agent':>10} {'Equal Wt':>10} {'EW B&H':>10}")
    print("-" * 66)
    for name, (a, e, b) in metrics.items():
        if name in ["Final Value($1)", "Sharpe Ratio", "Calmar Ratio"]:
            print(f"{name:<22} {a:>10.3f} {e:>10.3f} {b:>10.3f}")
        else:
            print(f"{name:<22} {a:>10.1%} {e:>10.1%} {b:>10.1%}")
    print("=" * 66)

    agent_hit = (agent_r > equal_r).mean()
    equal_hit = (equal_r > agent_r).mean()
    print(f"{'Hit Rate (Agent vs Eq)':<22} {agent_hit:>10.1%} {equal_hit:>10.1%} {'-':>10}")


# Plot

def plot_results(agent_r, equal_r, ew_bh_r, weights_df, ticker_names=None, plot = True):

    if not plot:
        return

    agent_cum = (1 + agent_r).cumprod()
    equal_cum = (1 + equal_r).cumprod()
    ew_bh_cum = (1 + ew_bh_r).cumprod()

    # Drawdown series
    agent_dd  = (agent_cum - agent_cum.cummax()) / agent_cum.cummax()
    equal_dd  = (equal_cum - equal_cum.cummax()) / equal_cum.cummax()

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)


    # 1. Cumulative Returns 
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(agent_cum.values, label="PPO Agent",     color="steelblue", lw=2)
    ax1.plot(equal_cum.values, label="Equal Weight",  color="tomato",    lw=2, ls="--")
    ax1.plot(ew_bh_cum.values, label="EW Buy & Hold", color="#2A9D8F", lw=2, ls=":")
    ax1.set_title("Cumulative Returns: Agent vs Equal Weight vs EW Buy & Hold")
    ax1.set_ylabel("Portfolio Value ($1 start)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Drawdown 
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(range(len(agent_dd)), agent_dd.values, color="steelblue", alpha=0.4)
    ax2.fill_between(range(len(equal_dd)), equal_dd.values, color="tomato",    alpha=0.3)
    ax2.plot(agent_dd.values, color="steelblue", lw=1, label="Agent")
    ax2.plot(equal_dd.values, color="tomato",    lw=1, label="Equal Wt")
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Rolling hit rate
    ax3 = fig.add_subplot(gs[1,1])
    window = 21
    beat = (agent_r > equal_r).astype(float)
    roll_hit = beat.rolling(window = window, min_periods = max(5, window // 3)).mean()

    ax3.plot(roll_hit.values * 100, color = "seagreen", lw=2, label = f"rolling{window}D Hit Rate")
    ax3.axhline(50, color = "gray", lw = 0.8, ls = "--", label = "50%")
    ax3.set_title(f"Rolling{window}D Hit Rate")
    ax3.set_ylabel("Hit Rate(%)")
    ax3.set_ylim(0,100)
    ax3.legend()
    ax3.grid(alpha = 0.3)

    # Portfolio weights
    ax4 = fig.add_subplot(gs[2, :])
    n_stocks = weights_df.shape[1]
    labels   = ticker_names if ticker_names else [f"Stock {i+1}" for i in range(n_stocks)]
    ax4.stackplot(range(len(weights_df)),
                  [weights_df[i].values for i in range(n_stocks)],
                  labels=labels, alpha=0.8)
    ax4.set_title("Portfolio Weight Allocation Over Time")
    ax4.set_ylabel("Weight")
    ax4.set_ylim(0, 1)
    ax4.legend(loc="upper right", fontsize=7, ncol=min(n_stocks, 6))
    ax4.grid(alpha=0.2)

    plt.suptitle("PPO Portfolio Backtest", fontsize=14, fontweight="bold", y=1.02)
    plt.show()


