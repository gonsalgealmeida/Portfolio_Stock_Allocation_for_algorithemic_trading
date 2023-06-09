##(The project for fulfillment of an integral part of the Data incubator's fellowship program( 2023 - spring - cohort)
# Risk Analysis and Portfolio Stock Allocation For Algorithmic Trading:

# Project Summary
This project focuses on the application of algorithmic trading and machine learning in investment management. It aims to utilize machine learning algorithms to automate investment tasks, extract information from data, and optimize portfolio allocation. The project explores various applications of machine learning, including risk analysis, alpha factor generation, strategy aggregation, and asset allocation. The objective is to enhance investment decision-making, achieve superior risk-return characteristics, and generate alpha in the trading process.
Project Objectives:

1. Explore the integration of algorithmic trading and machine learning in investment management.

2. Utilize machine learning algorithms to automate investment tasks and improve decision-making.

3. Investigate risk analysis techniques using machine learning for more effective portfolio allocation.

4. Generate alpha through the development and evaluation of alpha factors using machine learning.

5. Analyze and optimize trading strategies through strategy aggregation and asset allocation techniques.

6. Evaluate the performance and effectiveness of machine learning-based trading strategies through testing   and validation.
By achieving these objectives, the project aims to contribute to the advancement of algorithmic trading and machine learning applications in investment management, leading to more implementation to improve investment outcomes and enhanced portfolio performance.
The following are four sections as followed the above objectives:
* [01 Market Data Sources  ]
* [02 Alpha_factor_evaluation]
* [03 Strategy_evaluation]
* [04 Portfolio_stock_allocation]
# 01 Data
## Market Data Sources 
This section focuses on market and fundamental data sources and techniques. It covers working with high-frequency market data and API access to market data. The Python notebook also covered into how to work with fundamental data and compared different formats for efficient data storage with pandas. 
There are several options to access market data via API using Python.

### pandas datareader
The notebook [01_pandas_datareader_demo](01_pandas_datareader_demo.ipynb) presents a few sources built into the pandas library. 
- The `pandas` library enables access to data displayed on websites using the read_html function 
- the related `pandas-datareader` library provides access to the API endpoints of various data providers through a standard interface. 
### yfinance

The notebook [yfinance_demo](02_yfinance_demo.ipynb) shows how to use yfinance to download a variety of data from Yahoo! Finance. The library works around the deprecation of the historical data API by scraping data from the website in a reliable, efficient way with a Pythonic API.

### Qandl

The notebook [03_quandl_demo](03_quandl_demo.ipynb) shows how Quandl uses a very straightforward API to make its free and premium data available. See [documentation](https://www.quandl.com/tools/api) for more details.
### zipline & Qantopian

The notebook [contains the notebook [zipline_data](05_zipline_data.ipynb) briefly introduces the backtesting library `zipline` that we will use throughout this project and show how to access stock price data while running a backtest. 
# 02 Alpha_factor_evaluation
              
This section is a guide to financial feature engineering and alpha factor research for algorithmic trading strategies. It defines alpha factors as signals that aim to produce uncorrelated returns and provides an overview of the tools used to compute and test alpha factors. The document emphasizes the importance of building on decades of factor research to identify new factors that capture risks better than known factors. The open-source zipline library is introduced as a tool for backtestings. 

# 03 Strategy_evaluation
These notebooks evaluate portfolio optimization and performance. It explains that to test a strategy before implementing it under market conditions, one needs to simulate the trades the algorithm would make and verify their performance. It covers various topics, including measuring portfolio performance using metrics that reflect the return and risk of the investment portfolio. The note books also covered how modern portfolio theory, diversification, mean-variance optimization, and the capital asset pricing model are used to manage portfolio risk and return.

The document also presents alternatives to mean-variance optimization, such as the 1/N portfolio, the minimum-variance portfolio, and the Black-Litterman approach. It also covers the Kelly rule and Hierarchical Risk Parity as novel approaches to optimize portfolios, including the application of machine learning to learn hierarchical relationships among assets and treat their holdings as complements or substitutes with respect to the portfolio risk profile. Finally, the document discusses Pyfolio, which facilitates the analysis of portfolio performance and risk in-sample and out-of-sample using many standard metrics.

# 04 Portfolio_stock_allocation
The notebook explores concepts such as modern portfolio theory, diversification, mean-variance optimization, and the capital asset pricing model as methods for managing portfolio risk and return. Additionally, alternatives to mean-variance optimization, such as the 1/N portfolio, minimum-variance portfolio, and Black-Litterman approach, are discussed.

Finally, the notebooks delve into Pyfolio, a tool that facilitates the analysis of portfolio performance and risk using standard metrics. It allows for in-sample and out-of-sample evaluation, aiding in the assessment of portfolio strategies.

Overall, the notebooks provide a comprehensive exploration of portfolio optimization, including traditional and alternative approaches, as well as the analysis of portfolio performance using Pyfolio.


Mean-variance optimization requires two things: the expected returns of the assets and the covariance matrix (or, more generally, a risk model quantifying asset risk). We used PyPortfolioOpt for estimating both (located in expected_returns and risk_models, respectively).
Processed historical prices of 20 stocks:
    
           ["MSFT", "AMZN", "KO", "MA", "COST", 
           "LUV", "XOM", "PFE", "JPM", "UNH", 
           "ACN", "DIS", "GILD", "F", "TSLA","NVDA","AAPL","AMD","F","SOS","RGS"] 

# Structure
```
README.md
|   
|       
+---01_Fundamental_data
|   
|
|               
+---02_Alpha_factor_evaluation
|   |   .DS_Store
|   |   01_feature_engineering.ipynb
|   |   02_how_to_use_talib.ipynb
|   |   03_kalman_filter_and_wavelets.ipynb
|   |   04_single_factor_zipline.ipynb
|   |   06_performance_eval_alphalens.ipynb
|   |   factor_data.csv
|   |   single_factor.pickle
|   |   
|   \---.ipynb_checkpoints
|           01_feature_engineering-checkpoint.ipynb
|           02_how_to_use_talib-checkpoint.ipynb
|           04_single_factor_zipline-checkpoint.ipynb
|           06_performance_eval_alphalens-checkpoint.ipynb
|           
+---03_Strategy_evaluation
|   |   .DS_Store
|   |   01_backtest_with_trades.ipynb
|   |   02_backtest_with_pf_optimization.ipynb
|   |   03_pyfolio_demo.ipynb
|   |   04_mean_variance_optimization.ipynb
|   |   05_kelly_rule.ipynb
|   |   backtests.h5
|   |   
|   +---.ipynb_checkpoints
|   |       01_backtest_with_trades-checkpoint.ipynb
|   |       02_backtest_with_pf_optimization-checkpoint.ipynb
|   |       04_mean_variance_optimization-checkpoint.ipynb
|   |       05_kelly_rule-checkpoint.ipynb
|   |       
|   \---data
|           spy_prices.csv
|           
+---04_Portfolio_stock_allocation
|   |   .DS_Store
|   |   1-RiskReturnModels.ipynb
|   |   2-Mean-Variance-Optimisation.ipynb
|   |   2-Mean-Variance-Optimisation2.ipynb
|   |   2-Mean-Variance-Optimisation3.ipynb
|   |   3-Advanced-Mean-Variance-Optimisation.ipynb
|   |   4-Black-Litterman-Allocation.ipynb
|   |   5-Hierarchical-Risk-Parity.ipynb
|   |   Algorithemic_trading_with_Django_API_data_access2.ipynb
|   |   Portfolio_Allocation2.py
|   |   Portfolio_stock_Allocation2.ipynb
|   |   Portfolio_Allocation_API_dataaccess.ipynb
|   |   Portfolio_Allocation.py
|   |   processed_data.csv
|   |   stock_analysis.ipynb
|   |   
|   +---.ipynb_checkpoints
|   |       1-RiskReturnModels-checkpoint.ipynb
|   |       2-Mean-Variance-Optimisation-checkpoint.ipynb
|   |       2-Mean-Variance-Optimisation2-checkpoint.ipynb
|   |       2-Mean-Variance-Optimisation3-checkpoint.ipynb
|   |       3-Advanced-Mean-Variance-Optimisation-checkpoint.ipynb
|   |       4-Black-Litterman-Allocation-checkpoint.ipynb
|   |       5-Hierarchical-Risk-Parity-checkpoint.ipynb
|   |       Algorithemic_trading_with_Django_API_data_access2-checkpoint.ipynb
|   |       Portfolio_Allocation_API_dataaccess-checkpoint.ipynb
|   |       
|   +---data
|   |       spy_prices.csv
|   |       stock_prices.csv
|   |       
|   +---stocks
|   |   |   .DS_Store
|   |   |   db.sqlite3
|   |   |   manage.py
|   |   |   
|   |   +---prices
|   |   |   |   .DS_Store
|   |   |   |   admin.py
|   |   |   |   apps.py
|   |   |   |   models.py
|   |   |   |   tests.py
|   |   |   |   views.py
|   |   |   |   __init__.py
|   |   |   |   
|   |   |   +---migrations
|   |   |   |       __init__.py
|   |   |   |       
|   |   |   \---__pycache__
|   |   |           models.cpython-39.pyc
|   |   |           views.cpython-39.pyc
|   |   |           __init__.cpython-39.pyc
|   |   |           
|   |   \---stocks
|   |       |   .DS_Store
|   |       |   asgi.py
|   |       |   settings.py
|   |       |   urls.py
|   |       |   wsgi.py
|   |       |   __init__.py
|   |       |   
|   |       \---__pycache__
|   |               settings.cpython-39.pyc
|   |               urls.cpython-39.pyc
|   |               wsgi.cpython-39.pyc
|   |               __init__.cpython-39.pyc
|   |               
|   \---__pycache__
|           API_access.cpython-39.pyc
|           API_create.cpython-39.pyc
|           API_data.cpython-39.pyc
|           
+---data
|   |   .DS_Store
|   |   bbc.zip
|   |   create_datasets.ipynb
|   |   create_stooq_data.ipynb
|   |   create_yelp_review_data.ipynb
|   |   earnings_calls.zip
|   |   glove_word_vectors.ipynb
|   |   README.md
|   |   us_equities_meta_data.csv
|   |   wiki_stocks.csv
|   |   ^spx_d.csv
|   |   
|   +---.ipynb_checkpoints
|   |       create_datasets-checkpoint.ipynb
|   |       create_stooq_data-checkpoint.ipynb
|   |       
|   \---stooq
|           ^spx_d (2).csv
|           
\---figures
    |   .DS_Store
    |   s4_1.png
    |   s4_10.png
    |   s4_11.png
    |   s4_12.png
    |   s4_13.png
    |   s4_2.png
    |   s4_3.pdf
    |   s4_3.png
    |   s4_4.png
    |   s4_5.png
    |   s4_6.png
    |   s4_7.png
    |   s4_8.png
    |   s4_9.png
    |   
    \---.ipynb_checkpoints
    ```


# Brief RESULTS

#The long/short portfolio
|	    ('AAPL'| 0.00287)| |
|:----|:----|:----|
|             ('ACN'| 0.21881)| |
|             ('AMD'| -0.02401)| |
|             ('AMZN'| 0.00992)| |
|             ('COST'| 0.09775)| |
|             ('DIS'| -0.01794)| |
|             ('F'| -0.01975)| |
|             ('GILD'| 0.04944)| |
|             ('JPM'| -0.04251)| |
|             ('KO'| 0.12884)| |
|             ('LUV'| 0.02335)| |
|             ('MA'| 0.21963)| |
|             ('MSFT'| 0.00847)| |
|             ('NVDA'| -0.01443)| |
|             ('PFE'| 0.07013)| |
|             ('RGS'| 0.01735)| |
|             ('SOS'| 0.04845)| |
|             ('TSLA'| 0.11261)| |
|             ('UNH'| 0.03353)| |
|             ('XOM'| 0.07748)|


Annual volatility: 12.0%
(Note that this is an in sample estimate and may have very little resemblance to how the portfolio actually performs!)

Assume, we had $20,000 to invest and would like our portfolio to be 130/30 long/short. 

#Constructed the actual allocation as follows:
||   |    |
|:----|:----|:----|
|             ('AAPL'| 1)||
|             ('ACN'| 13)| |
|             ('AMD'| -11)| |
|             ('AMZN'| 2)| |
|             ('COST'| 4)| |
|             ('DIS'| -10)| |
|             ('F'| -86)| |
|             ('GILD'| 11)| |
|             ('JPM'| -16)| |
|             ('KO'| 3)| |
|             ('LUV'| 14)| |
|             ('MA'| 10)| |
|             ('MSFT'| 2)| |
|             ('NVDA'| -2)| |
|             ('PFE'| 32)| |
|             ('RGS'| 307)| |
|             ('SOS'| 216)| |
|             ('TSLA'| 11)| |
|             ('UNH'| 1)| |
|             ('XOM'| 13)]|


#Constructed the portfolio allocation( maximizing return for a given volatility=15%,with L2 regularization)
 
|	    ('AAPL'| 0.03212)| |
|:----|:----|:----|
|             ('ACN'| 0.06016)| |
|             ('AMD'| 0.02457)| |
|             ('AMZN'| 0.05828)| |
|             ('COST'| 0.0524)| |
|             ('DIS'| 0.02439)| |
|             ('F'| 0.01999)| |
|             ('GILD'| 0.04772)| |
|             ('JPM'| 0.02607)| |
|             ('KO'| 0.0476)| |
|             ('LUV'| 0.03057)| |
|             ('MA'| 0.08082)| |
|             ('MSFT'| 0.03613)| |
|             ('NVDA'| 0.06298)| |
|             ('PFE'| 0.03603)| |
|             ('RGS'| 0.03943)| |
|             ('SOS'| 0.17097)| |
|             ('TSLA'| 0.08874)| |
|             ('UNH'| 0.03853)| |
|             ('XOM'| 0.0225)|


![s4_7](https://github.com/gonsalgealmeida/Portfolio_Stock_Allocation_for_algorithemic_trading/assets/49290976/f6567f0f-6e6b-4651-ae87-9e1a5e02f21e)


![]('/Users/surekaalmeida/Documents/dataincubator/TDI_Capstone/s4_7.png')
Expected annual return: 31.6%
Annual volatility: 15.0%
Sharpe Ratio: 1.97.

#Constructed portpolio with minimal risk for 7% return.
|[('AAPL'| 0.0026)| |
|:----|:----|:----|
|             ('ACN'| -0.0141)| |
|             ('AMD'| 0.02232)| |
|             ('AMZN'| 0.01807)| |
|             ('COST'| -0.01903)| |
|             ('DIS'| -0.01037)| |
|             ('F'| -0.00937)| |
|             ('GILD'| -0.00791)| |
|             ('JPM'| -0.00453)| |
|             ('KO'| -0.02938)| |
|             ('LUV'| -0.01077)| |
|             ('MA'| -0.00182)| |
|             ('MSFT'| -0.0055)| |
|             ('NVDA'| 0.03074)| |
|             ('PFE'| -0.02269)| |
|             ('RGS'| -0.00565)| |
|             ('SOS'| 0.10444)| |
|             ('TSLA'| 0.00805)| |
|             ('UNH'| -0.01451)| |
|             ('XOM'| -0.03057)])|

Expected annual return: 7.0%
Annual volatility: 5.5%
Sharpe Ratio: 0.91
![s4_8](https://github.com/gonsalgealmeida/Portfolio_Stock_Allocation_for_algorithemic_trading/assets/49290976/72ded450-9555-4dda-bbf6-3a1f186f72b6)

# Discussion
The above predictions were generated by employing Mean-variance optimization methods and utilizing various machine learning (ML) models to optimize portfolios. The calculations require selecting stocks for the portfolio, which is based on major stock models and recommendations derived from investigating the stock cycle.

We strongly believe that by incorporating historical stock data and applying supervised learning algorithms, the stock selection process can be improved. Additionally, certain trading techniques can be utilized to provide insights into the properties of the models.

The results presented above were obtained using the following techniques: 
* Calculating and visualising the covariance matrix. 

* Optimising a long/short portfolio to minimise total variance.

* Optimising a portfolio to maximise the Sharpe ratio, subject to sector constraints.

* Optimising a portfolio to maximise return for a given risk, subject to sector constraints.     with an L2 regularisation objective

* Optimising a market-neutral portfolio to minimise risk for a given level of return.

By employing these techniques, the project aims to enhance portfolio optimization and achieve better performance in stock selection based on the investigation of stock cycles and the application of ML models.
# Alternative Algorithmic Trading Libraries and Platforms

- [QuantConnect](https://www.quantconnect.com/)
- [WorldQuant](https://www.worldquantvrc.com/en/cms/wqc/home/)
- Python Algorithmic Trading Library [PyAlgoTrade](http://gbeced.github.io/pyalgotrade/)
- [pybacktest](https://github.com/ematvey/pybacktest)
- [Trading with Python](http://www.tradingwithpython.com/)
- [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=5041)
- matplotlib [docs](https://github.com/matplotlib/matplotlib)
- numpy [docs](https://github.com/numpy/numpy)
- pandas [docs](https://github.com/pydata/pandas)
- scipy [docs](https://github.com/scipy/scipy)
- scikit-learn [docs](https://scikit-learn.org/stable/user_guide.html)
- LightGBM [docs](https://lightgbm.readthedocs.io/en/latest/)
- CatBoost [docs](https://catboost.ai/docs/concepts/about.html)
- TensorFlow [docs](https://www.tensorflow.org/guide)
- PyTorch [docs](https://pytorch.org/docs/stable/index.html)
- Machine Learning Financial Laboratory (mlfinlab) [docs]     (https://mlfinlab.readthedocs.io/en/latest/)
- seaborn [docs](https://github.com/mwaskom/seaborn)
- statsmodels [docs](https://github.com/statsmodels/statsmodels)
- [Boosting numpy: Why BLAS Matters](http://markus-beuckelmann.de/blog/boosting-numpy-   blas.html)


# For future Projects 
In future projects, the main objective will be to establish a solid foundation for algorithmic trading. The focus will be on investigating libraries and frameworks that can be utilized to implement the foundational models based on stochastic Levy processes.

The project will involve rigorous testing of machine learning models and various risk analysis methods. Once these models and methods have been thoroughly evaluated and validated, the next step will be to advance the project to the next level by incorporating the application of Levy processes.

The implementation of Levy processes will provide a more advanced and sophisticated approach to algorithmic trading. This will involve leveraging the principles and characteristics of Levy processes to enhance the accuracy and effectiveness of the trading models and strategies developed.

By incorporating the use of Levy processes, future projects can further optimize and refine algorithmic trading strategies, potentially improving risk management, return generation, and overall performance in the financial markets.


