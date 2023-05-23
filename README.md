##(The project for fullfilment of an integral part of the Data incubator's fellowship program( 2023-spring - cohor)
# Risk Analysis and Portfolio Stock Allocation for algorithemic      trading:

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
This section focus on market and fundamental data sources and techniques. It covers working with high-frequency market data, and API access to market data. The python note book also covered into how to work with fundamental data and compares different formats for efficient data storage with pandas. 
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
These notebooks evaluate portfolio optimization and performance. It explains that to test a strategy before implementing it under market conditions, one needs to simulate the trades the algorithm would make and verify their performance. It covers various topics, including measuring portfolio performance using metrics that reflect the return and risk of the investment portfolio. The note bookes also covered how modern portfolio theory, diversification, mean-variance optimization, and the capital asset pricing model are used to manage portfolio risk and return.

The document also presents alternatives to mean-variance optimization, such as the 1/N portfolio, the minimum-variance portfolio, and the Black-Litterman approach. It also covers the Kelly rule and Hierarchical Risk Parity as novel approaches to optimize portfolios, including the application of machine learning to learn hierarchical relationships among assets and treat their holdings as complements or substitutes with respect to the portfolio risk profile. Finally, the document discusses Pyfolio, which facilitates the analysis of portfolio performance and risk in-sample and out-of-sample using many standard metrics.
              
              
These notebooks focus on evaluating portfolio optimization and performance. They emphasize the importance of simulating trades and testing strategies before implementing them in real market conditions. Various topics are covered, including measuring portfolio performance using metrics that consider both return and risk.
# 04 Portfolio_stock_allocation

The notebooks explore concepts such as modern portfolio theory, diversification, mean-variance optimization, and the capital asset pricing model as methods for managing portfolio risk and return. Additionally, alternatives to mean-variance optimization, such as the 1/N portfolio, minimum-variance portfolio, and Black-Litterman approach, are discussed.

Finally, the notebooks delve into Pyfolio, a tool that facilitates the analysis of portfolio performance and risk using standard metrics. It allows for in-sample and out-of-sample evaluation, aiding in the assessment of portfolio strategies.

Overall, the notebooks provide a comprehensive exploration of portfolio optimization, including traditional and alternative approaches, as well as the analysis of portfolio performance using Pyfolio.

This note books evaluate on mean-variance optimization (MVO), which is what most people think of when they hear “portfolio optimization”.

Mean-variance optimization requires two things: the expected returns of the assets, and the covariance matrix (or more generally, a risk model quantifying asset risk). we used PyPortfolioOpt used methods for estimating both (located in expected_returns and risk_models respectively).
Processed historical prices of 20 stocks:
    
           ["MSFT", "AMZN", "KO", "MA", "COST", 
           "LUV", "XOM", "PFE", "JPM", "UNH", 
           "ACN", "DIS", "GILD", "F", "TSLA","NVDA","AAPL","AMD","F","SOS","RGS"] 

# The Whole Project Outline


# Brief RESULTS

#The long/short portfolio
||   |    |
|:----|:----|:----|
|             ('AAPL'| 0.00287)||
|             ('ACN'| 0.21881)| |
|             ('AMD'| -0.024)| |
|             ('AMZN'| 0.0099)| |
|             ('COST'| 0.09772)| |
|             ('DIS'| -0.01793)| |
|             ('F'| -0.01975)| |
|             ('GILD'| 0.04943)| |
|             ('JPM'| -0.04252)| |
|             ('KO'| 0.12877)| |
|             ('LUV'| 0.02334)| |
|             ('MA'| 0.2196)| |
|             ('MSFT'| 0.00848)| |
|             ('NVDA'| -0.01444)| |
|             ('PFE'| 0.07025)| |
|             ('RGS'| 0.01736)| |
|             ('SOS'| 0.04845)| |
|             ('TSLA'| 0.11267)| |
|             ('UNH'| 0.03353)| |
|             ('XOM'| 0.07744)]|

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


#Constructed the portfolio allocation( maximing return for a given volatility=15%,with L2 regularisation)
 
|[('AAPL'| 0.03212)| |
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
|             ('XOM'| 0.0225)])|

#s4_7.png
import os
cwd=os.getcwd()
cwd
![]('/Users/surekaalmeida/Documents/dataincubator/TDI_Capstone/s4_7.png')
Expected annual return: 31.6%
Annual volatility: 15.0%
Sharpe Ratio: 1.97.

#Constructed fortpolio with minimise risk for 7% return.
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
s4_8.png

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


