import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import yfinance as yf
from pypfopt import risk_models, plotting, EfficientFrontier, DiscreteAllocation, expected_returns

_all_=['download_data','calculate_covariance','portfolio_optimization','discrete_allocation','apply_sector_constraints','plot_weights','print_total_weights','add_objective_function','market_neutral_portfolio','efficient_frontier_semi_covariance','efficient_frontier_semi_covariance','efficient_semivariance','compute_var_and_cvar','plot_efficient_frontier','plot_efficient_frontier','add_constraint_and_plot','monte_carlo_simulation']

def download_data(tickers):
    ohlc = yf.download(tickers, period="max")
    prices = ohlc["Adj Close"].dropna(how="all")
    return prices

def calculate_covariance(prices):
    sample_cov = risk_models.sample_cov(prices, frequency=252)
    return sample_cov

def portfolio_optimization(prices):
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(None, S, weight_bounds=(None, None))
    ef.min_volatility()
    weights = ef.clean_weights()
    return weights, ef

def discrete_allocation(weights, prices, portfolio_value):
    latest_prices = prices.iloc[-1]  
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_value, short_ratio=0.3)
    alloc, leftover = da.lp_portfolio()
    return alloc, leftover

def apply_sector_constraints(ef, sector_mapper, sector_lower, sector_upper, tickers):
    mu = expected_returns.capm_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    ef.add_constraint(lambda w: w[tickers.index("AMZN")] == 0.10)
    ef.add_constraint(lambda w: w[tickers.index("TSLA")] <= 0.05)
    ef.add_constraint(lambda w: w[10] >= 0.05)
    ef.max_sharpe()
    weights = ef.clean_weights()
    return weights

def plot_weights(weights):
    pd.Series(weights).plot.pie(figsize=(10,10))
    plt.show()
def print_total_weights(weights, sector_mapper):
    for sector in set(sector_mapper.values()):
        total_weight = sum(w for t, w in weights.items() if sector_mapper[t] == sector)
        print(f"{sector}: {total_weight:.3f}")

def add_objective_function(ef, gamma):
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)  # gamma is the tuning parameter

def market_neutral_portfolio(mu, S):
    ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
    ef.add_objective(objective_functions.L2_reg)
    ef.efficient_return(target_return=0.07, market_neutral=True)
    weights = ef.clean_weights()
    return weights

def efficient_frontier_semi_covariance(mu, prices):
    semicov = risk_models.semicovariance(prices, benchmark=0)
    ef = EfficientFrontier(mu, semicov)
    ef.efficient_return(0.2)
    weights = ef.clean_weights()
    return weights

def efficient_semivariance(mu, returns):
    es = EfficientSemivariance(mu, returns)
    es.efficient_return(0.2)
    es.portfolio_performance(verbose=True)

def compute_var_and_cvar(returns, weight_arr):
    portfolio_rets = (returns * weight_arr).sum(axis=1)
    var = portfolio_rets.quantile(0.05)
    cvar = portfolio_rets[portfolio_rets <= var].mean()
    print("VaR: {:.2f}%".format(100*var))
    print("CVaR: {:.2f}%".format(100*cvar))

def plot_efficient_frontier(mu, S):
    cla = CLA(mu, S)
    ax = plotting.plot_efficient_frontier(cla, showfig=False)

def add_constraint_and_plot(mu, S, tickers):
    ef = EfficientFrontier(mu, S)
    big_tech_indices = [t in {"MSFT", "AMZN", "TSLA"} for t in tickers]
    ef.add_constraint(lambda w: cp.sum(w[big_tech_indices]) <= 0.3)
    ax = plotting.plot_efficient_frontier(ef, ef_param="risk", 
                                          ef_param_range=np.linspace(0.12, 0.4, 50), 
                                          showfig=False)

def monte_carlo_simulation(mu, S):
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
    sharpes = rets / stds

   